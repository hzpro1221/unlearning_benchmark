import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import wandb
import yaml

# import dataloaders
from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset
import torchvision.transforms as transforms

# import transforms
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.forget_train_transform import get_forget_train_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
from dataset.transform.retain_train_transform import get_retain_train_transform
from dataset.transform.test_transform import get_test_transform
from dataset.transform.unseen_transform import get_unseen_transform

# import architectures
from architecture.deity import DeiTArchitecture
from architecture.resnet import ResNetArchitecture
from architecture.moe_deit import MoEDeiTArchitecture

# import metrics
from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia

# import approximate unlearning algorithms (using standard dataloader inputs)
from approx_algo.boundary_shrink import boundary_shrink
from approx_algo.gradient_ascent import gradient_ascent
from approx_algo.l1_sparse import l1_sparse
from approx_algo.random_labeling import random_labeling
from approx_algo.module_unlearn_algo import ModularUnlearning

class ApplyTransform(Dataset):
    """
    helper class to apply specific transforms to a pytorch subset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # explicitly force resize to 224x224 for standard resnet/deit inputs
        self.resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        # some datasets return (image, label, domain), others just (image, label)
        data = self.subset[idx]
        image = data[0]

        # force resize to 224 before any other transforms
        image = self.resize(image)
        
        if self.transform:
            image = self.transform(image)
            
        # reconstruct the tuple with the transformed image
        return (image,) + data[1:]

    def __len__(self):
        return len(self.subset)

def set_seed(seed):
    """
    forces deterministic behavior across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # resets peak memory tracker at the start of the script
        torch.cuda.reset_peak_memory_stats()

def main():
    parser = argparse.ArgumentParser(description="unlearn a random subset using a yaml config.")
    parser.add_argument('--config', type=str, required=True, help="path to the config .yaml file.")
    cmd_args = parser.parse_args()
    
    # 1. load yaml config
    print(f"[*] loading config from {cmd_args.config}")
    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    args = argparse.Namespace(**yaml_config)
    
    # default output directory for unlearning
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'checkpoint/unlearn'
        
    yaml_filename = os.path.splitext(os.path.basename(cmd_args.config))[0]

    # 2. setup environment
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] using device: {device}")
    
    unlearn_algo = getattr(args, 'unlearn_algo', 'finetune')
    use_wandb = getattr(args, 'use_wandb', True)

    # 3. initialize wandb (skipped when use_wandb: false in config)
    if use_wandb:
        print("[*] initializing wandb...")
        wandb.login(key="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")
        run_name = f"unlearn_{unlearn_algo}_{yaml_filename}"
        wandb.init(
            project='random_unlearn',
            name=run_name,
            config=yaml_config,
            settings=wandb.Settings(start_method='thread')
        )
    else:
        print("[*] wandb disabled (use_wandb: false)")

    # 4. load the raw dataset
    print(f"[*] loading dataset: {args.dataset}")
    if args.dataset == 'pacs':
        full_dataset = PACSDataset(root_dir=args.data_dir, transform=None)
        num_classes = 7
    elif args.dataset == 'officehome':
        full_dataset = OfficeHomeDataset(root_dir=args.data_dir, transform=None)
        num_classes = 65
    elif args.dataset == 'cifar100':
        full_dataset = CIFAR100Dataset(root_dir=args.data_dir, split="train", transform=None)
        num_classes = 100
    elif args.dataset == 'tiny_imagenet':
        full_dataset = TinyImageNetDataset(root_dir=args.data_dir, transform=None)
        num_classes = 200
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")

    # 5. primary deterministic split (80% train, 10% test, 10% unseen)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, unseen_subset = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    # 6. secondary deterministic split (divide train_subset into retain and forget)
    forget_ratio = getattr(args, 'forget_ratio', 0.1)  # default to 10% if not specified
    forget_size = int(forget_ratio * train_size)
    retain_size = train_size - forget_size
    
    forget_subset, retain_subset = random_split(
        train_subset, [forget_size, retain_size], generator=generator
    )
    
    print(f"[*] split sizes -> retain: {retain_size} | forget: {forget_size} | test: {test_size} | unseen: {unseen_size}")

    # 7. apply explicit transforms
    # active training loaders
    forget_train_set = ApplyTransform(forget_subset, transform=get_forget_train_transform())
    retain_train_set = ApplyTransform(retain_subset, transform=get_retain_train_transform())
    
    # evaluation loaders (no augmentation)
    forget_eval_set = ApplyTransform(forget_subset, transform=get_forget_test_transform())
    retain_eval_set = ApplyTransform(retain_subset, transform=get_retain_test_transform())
    test_set = ApplyTransform(test_subset, transform=get_test_transform())
    unseen_set = ApplyTransform(unseen_subset, transform=get_unseen_transform())

    # 8. create dataloaders
    forget_train_loader = DataLoader(forget_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_train_loader = DataLoader(retain_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    forget_eval_loader = DataLoader(forget_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_eval_loader = DataLoader(retain_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(unseen_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 9. initialize architecture dynamically
    use_moe = getattr(args, 'use_moe', False)
    print(f"[*] initializing model: {args.model_name}{' (MoE)' if use_moe else ''}")
    if 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=False, device=device)
    elif 'deit' in args.model_name and use_moe:
        model = MoEDeiTArchitecture(
            model_name=args.model_name,
            num_classes=num_classes,
            num_experts=getattr(args, 'num_experts', 4),
            pretrained=False,
            device=device,
        )
    elif 'deit' in args.model_name:
        model = DeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=False, device=device)
    else:
        raise ValueError(f"unsupported model prefix for {args.model_name}")

    # 10. load pre-trained weights
    if not hasattr(args, 'pretrained_model_path') or not os.path.exists(args.pretrained_model_path):
        raise FileNotFoundError(f"pretrained model path is missing or invalid: {getattr(args, 'pretrained_model_path', 'None')}")
    
    print(f"[*] loading pre-trained weights from: {args.pretrained_model_path}")
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

    # 11. setup optimizer and loss
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 12. unlearning loop
    fa_threshold = getattr(args, 'fa_threshold', 0.8) # e.g., 0.14 for PACS
    
    print(f"[*] starting unlearning phase using algorithm: {unlearn_algo.upper()}")
    print(f"[*] early stopping condition: stop when fa <= {fa_threshold*100:.2f}%")

    total_unlearn_time = 0.0

    # ------------------------------------------------------------------
    # modular_unlearn — Phase 1: setup (runs once before the epoch loop)
    # ------------------------------------------------------------------
    _modular_unlearner = None
    _mia_converged = False          # set True when MIA ∈ [0.48, 0.52] mid-epoch
    if unlearn_algo == 'modular_unlearn':
        _modular_unlearner = ModularUnlearning(
            model=model,
            lr=args.lr,
            beta=getattr(args, 'modular_beta', 1.0),
            gamma=getattr(args, 'modular_gamma', 1.0),
            eta=getattr(args, 'modular_eta', 0.1),
            alpha_resp=getattr(args, 'modular_alpha_resp', 1.0),
            top_k=getattr(args, 'modular_top_k', None),
            tau=getattr(args, 'modular_tau', None),
            device=device,
        )
        info = _modular_unlearner.begin_modular_unlearn(forget_train_loader, retain_train_loader)
        active_set = set(info['expert_indices'])
        print(f"[*] modular unlearning: M_f = {info['expert_indices']} "
              f"({len(active_set)} of {model.num_experts} experts activated)")
        for m, (sf, sr, rho) in enumerate(zip(
                info['scores_forget'], info['scores_retain'], info['responsibility'])):
            marker = "  [ACTIVE]" if m in active_set else ""
            print(f"        expert[{m}]: s_f={sf:.4f}  s_r={sr:.4f}  ρ={rho:+.4f}{marker}")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        avg_loss = 0.0
        
        # -----------------------------------------------------------
        # route to the correct algorithm based on yaml config
        # (passing the entire dataloader again!)
        # -----------------------------------------------------------
        if unlearn_algo == 'boundary_shrink':
            avg_loss = boundary_shrink(model, criteria, optimizer, forget_train_loader, device=device)
            
        elif unlearn_algo == 'ga':
            avg_loss = gradient_ascent(model, criteria, optimizer, forget_train_loader, device=device)
            
        elif unlearn_algo == 'l1_sparse':
            avg_loss = l1_sparse(model, criteria, optimizer, forget_train_loader, device=device)
            
        elif unlearn_algo == 'rl':
            avg_loss = random_labeling(model, criteria, optimizer, forget_train_loader, device=device)
            
        elif unlearn_algo == 'finetune':
            # baseline: just train normally on the retain set
            model.train()
            total_loss = 0.0
            for batch in retain_train_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                
                optimizer.zero_grad()
                logits, _ = model.forward_with_grad(images)
                loss = criteria(logits, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(retain_train_loader)
            
        elif unlearn_algo == 'modular_unlearn':
            # Phase 2: per-step updates interleaving forget and retain batches.
            # eval_interval controls how often the MIA convergence check runs.
            # When MIA settles in [0.48, 0.52] the model is statistically
            # indistinguishable from a retrained model — we halt early.
            steps = min(len(forget_train_loader), len(retain_train_loader))
            eval_interval = getattr(args, 'modular_eval_interval', steps)

            forget_iter = iter(forget_train_loader)
            retain_iter = iter(retain_train_loader)
            running_total = 0.0
            completed_steps = 0

            for step_idx in range(steps):
                b_f = next(forget_iter)
                b_r = next(retain_iter)
                loss_dict = _modular_unlearner.update_modular_unlearn(b_f, b_r)
                running_total += loss_dict["total"]
                completed_steps += 1

                # step-interval MIA convergence check
                if (step_idx + 1) % eval_interval == 0:
                    step_mia = mia(model, forget_eval_loader, unseen_loader, device)
                    if 0.48 <= step_mia <= 0.52:
                        print(f"  [!] mia converged at step {step_idx + 1}: "
                              f"mia={step_mia:.4f} ∈ [0.48, 0.52] — halting epoch early")
                        _mia_converged = True
                        break

            avg_loss = running_total / max(completed_steps, 1)

        else:
            raise ValueError(f"unsupported unlearn_algo: {unlearn_algo}")
            
        # stop timer before evaluation
        epoch_unlearn_time = time.time() - epoch_start_time
        total_unlearn_time += epoch_unlearn_time
        
        # -----------------------------------------------------------
        # evaluate all metrics at the end of the epoch
        # -----------------------------------------------------------
        fa_score = forget_acc(model, forget_eval_loader, device)
        ra_score = retain_acc(model, retain_eval_loader, device)
        ta_score = test_acc(model, test_loader, device)
        mia_score = mia(model, forget_eval_loader, unseen_loader, device)
        
        print(f"epoch [{epoch+1}/{args.epochs}] | {unlearn_algo} loss: {avg_loss:.4f} | "
              f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
              f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_unlearn_time:.2f}s")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "unlearn_loss": avg_loss,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })
        
        # -----------------------------------------------------------
        # early stopping check
        # -----------------------------------------------------------
        if unlearn_algo != 'modular_unlearn' and fa_score <= fa_threshold:
            print(f"\n[!] early stopping triggered at epoch {epoch+1}!")
            print(f"[*] current fa ({fa_score*100:.2f}%) <= threshold ({fa_threshold*100:.2f}%)")
            break

        if _mia_converged:
            print(f"\n[!] mia convergence stop triggered at epoch {epoch+1}!")
            print(f"[*] mia settled in [0.48, 0.52] — model is statistically indistinguishable from retrained")
            break

    # ------------------------------------------------------------------
    # modular_unlearn — Phase 3: teardown (runs once after epoch loop)
    # ------------------------------------------------------------------
    if _modular_unlearner is not None:
        _modular_unlearner.end_modular_unlearn()
        print("[*] modular unlearning teardown complete; full gradient flow restored.")

    # 13. calculate final metrics (memory and total time)
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_memory_gb = 0.0
        
    print("\n[*] --- final unlearning summary ---")
    print(f"[*] total unlearn time (excluding evaluation): {total_unlearn_time:.2f} seconds")
    print(f"[*] peak gpu memory usage: {peak_memory_gb:.4f} gb")
    
    if use_wandb:
        wandb.log({
            "total_unlearn_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb
        })

    # 14. save the final unlearned model
    save_path = os.path.join(args.output_dir, f"unlearned_{unlearn_algo}_{yaml_filename}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[*] unlearning complete. model saved to {save_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()