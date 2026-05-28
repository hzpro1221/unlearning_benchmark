import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import wandb
import yaml
import torchvision.transforms as transforms

# import dataloaders
from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset

# import transforms
from dataset.transform.train_transform import get_train_transform
from dataset.transform.test_transform import get_test_transform
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
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

# ------------------------------------------------------------------
# MoE auxiliary training losses
# ------------------------------------------------------------------

def _compute_sp_loss(moe_adapters) -> torch.Tensor:
    """
    Routing sparsity loss — minimise routing entropy so each sample
    activates a small subset of experts.

        L_sp = E_x[ -Σ_m π_m(x) log π_m(x) ]

    Accumulated and averaged over all transformer blocks.
    Gradients flow through adapter.last_routing_weights → router params.
    """
    total, count = None, 0
    for a in moe_adapters:
        if a.last_routing_weights is None:
            continue
        pi = a.last_routing_weights                          # (B, T, M), with grad
        entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)     # (B, T)
        block_sp = entropy.mean()
        total = block_sp if total is None else total + block_sp
        count += 1
    return total / count if count > 0 else torch.tensor(0.0)


def _compute_bal_loss(moe_adapters, ema_pi: torch.Tensor, ema_alpha: float):
    """
    Routing balance loss — prevent routing collapse by penalising
    deviation of per-expert usage from the uniform target 1/M.

        L_bal = Σ_m (π̂_m - 1/M)²

    π̂_m is an EMA estimate of mean routing probability for expert m:
        π̂_m^(t) = α · π̂_m^(t-1).detach() + (1-α) · batch_mean_m

    The (1-α) term retains gradient so the loss differentiates through
    the current batch's routing probabilities → router params.
    The α · prev term is detached so gradients don't accumulate across steps.

    Returns:
        l_bal     — scalar loss tensor (with grad)
        new_ema   — updated EMA tensor (detached, for use in next step)
    """
    M = ema_pi.shape[0]
    pis = [a.last_routing_weights for a in moe_adapters if a.last_routing_weights is not None]
    if not pis:
        return torch.tensor(0.0), ema_pi

    # Average over blocks, batch, and token positions → (M,), with grad
    stacked = torch.stack(pis, dim=0)           # (num_blocks, B, T, M)
    batch_mean = stacked.mean(dim=(0, 1, 2))    # (M,)

    # EMA update: only the (1-alpha) * batch_mean term carries gradient
    new_ema = ema_alpha * ema_pi.detach() + (1.0 - ema_alpha) * batch_mean
    l_bal = ((new_ema - 1.0 / M) ** 2).sum()

    return l_bal, new_ema.detach()


def _compute_div_loss(moe_adapters, eps: float = 1e-8) -> torch.Tensor:
    """
    Expert specialisation loss — penalise cross-expert output correlation.

        L_div = Σ_{m≠n} ‖(1/B) H̃_m^T H̃_n‖_F²

    where H̃_m = H_m / (‖H_m‖_F + ε), H_m ∈ R^{B×D} is the token-averaged
    output of expert m over the current mini-batch.
    Accumulated over all transformer blocks.
    Gradients flow through adapter.last_expert_outputs → expert params.
    """
    total = None
    for a in moe_adapters:
        if a.last_expert_outputs is None:
            continue
        H = a.last_expert_outputs       # (B, M, D), with grad
        B, M, D = H.shape

        # Per-expert Frobenius norm: ‖H_m‖_F where H_m ∈ R^{B×D}
        # = sqrt(Σ_b Σ_d H[b,m,d]²) vectorised over M
        norms = H.pow(2).sum(dim=2).sum(dim=0).sqrt().clamp(min=eps)  # (M,)
        H_n = H / norms.view(1, -1, 1)  # (B, M, D) normalised

        block_div = H.new_zeros(())
        for m in range(M):
            for n in range(m + 1, M):
                cross = (H_n[:, m, :].T @ H_n[:, n, :]) / B  # (D, D)
                block_div = block_div + cross.pow(2).sum()
        block_div = block_div * 2   # count both (m,n) and (n,m)

        total = block_div if total is None else total + block_div

    return total if total is not None else torch.tensor(0.0)


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

def get_domain(dataset, idx):
    """helper to extract domain label safely"""
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    else:
        data_tuple = dataset[idx]
        domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
        return int(domain_val)

def main():
    parser = argparse.ArgumentParser(description="train a base model from yaml config.")
    parser.add_argument('--config', type=str, required=True, help="path to the config .yaml file.")
    cmd_args = parser.parse_args()
    
    # 1. load yaml config
    print(f"[*] loading config from {cmd_args.config}")
    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    args = argparse.Namespace(**yaml_config)
    
    # the output directory is set fixed
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'checkpoint/learn'
        
    yaml_filename = os.path.splitext(os.path.basename(cmd_args.config))[0]

    # 2. setup environment
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] using device: {device}")
    
    use_wandb = getattr(args, 'use_wandb', True)

    # 3. initialize wandb (skipped when use_wandb: false in config)
    if use_wandb:
        print("[*] initializing wandb...")
        wandb.login(key="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")
        run_name = f"base_{yaml_filename}"
        wandb.init(
            project='learn',
            name=run_name,
            config=yaml_config,
            settings=wandb.Settings(start_method='thread')
        )
    else:
        print("[*] wandb disabled (use_wandb: false)")

    # 4. load the raw dataset (without transforms yet)
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

    # 5. primary deterministic split using random_split (80% train, 10% test, 10% unseen)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, unseen_subset = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    # 6. secondary deterministic split based on unlearn setting
    unlearn_setting = getattr(args, 'unlearn_setting', 'random')
    print(f"[*] unlearn setting applied: {unlearn_setting.upper()}")
    
    if unlearn_setting == 'random':
        forget_ratio = getattr(args, 'forget_ratio', 0.1)
        forget_size = int(forget_ratio * train_size)
        retain_size = train_size - forget_size
        
        forget_subset, retain_subset = random_split(
            train_subset, [forget_size, retain_size], generator=generator
        )
        final_test_subset = test_subset
        
    elif unlearn_setting == 'class':
        forget_classes = getattr(args, 'forget_classes', [0])
        if not isinstance(forget_classes, list): forget_classes = [forget_classes]
        
        f_tr_idx, r_tr_idx, r_te_idx = [], [], []
        
        for idx in train_subset.indices:
            lbl = full_dataset.labels[idx]
            if lbl in forget_classes: 
                f_tr_idx.append(idx)
            else: 
                r_tr_idx.append(idx)
            
        for idx in test_subset.indices:
            lbl = full_dataset.labels[idx]
            if lbl not in forget_classes: 
                r_te_idx.append(idx)
            
        forget_subset = Subset(full_dataset, f_tr_idx)
        retain_subset = Subset(full_dataset, r_tr_idx)
        final_test_subset = Subset(full_dataset, r_te_idx)
        
    elif unlearn_setting == 'domain':
        if args.dataset not in ['pacs', 'officehome']:
            raise ValueError(f"domain unlearning not supported for {args.dataset}")
            
        forget_domains = getattr(args, 'forget_domains', [0])
        if not isinstance(forget_domains, list): forget_domains = [forget_domains]
        
        f_tr_idx, r_tr_idx, r_te_idx = [], [], []
        
        for idx in train_subset.indices:
            dom = get_domain(full_dataset, idx)
            if dom in forget_domains: 
                f_tr_idx.append(idx)
            else: 
                r_tr_idx.append(idx)
            
        for idx in test_subset.indices:
            dom = get_domain(full_dataset, idx)
            if dom not in forget_domains: 
                r_te_idx.append(idx)
            
        forget_subset = Subset(full_dataset, f_tr_idx)
        retain_subset = Subset(full_dataset, r_tr_idx)
        final_test_subset = Subset(full_dataset, r_te_idx)
        
    else:
        raise ValueError("unlearn_setting must be 'random', 'class', or 'domain'")

    print(f"[*] split sizes -> train full: {train_size} (retain eval: {len(retain_subset)} | forget eval: {len(forget_subset)})")
    print(f"[*] split sizes -> test: {len(final_test_subset)} | unseen: {unseen_size}")

    # 7. apply explicit transforms
    # active training uses the full train_subset with augmentations
    train_set = ApplyTransform(train_subset, transform=get_train_transform())
    
    # evaluation sets strictly use test transforms (no randomness)
    forget_eval_set = ApplyTransform(forget_subset, transform=get_forget_test_transform())
    retain_eval_set = ApplyTransform(retain_subset, transform=get_retain_test_transform())
    test_set = ApplyTransform(final_test_subset, transform=get_test_transform())
    unseen_set = ApplyTransform(unseen_subset, transform=get_unseen_transform())

    # 8. create dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_eval_loader = DataLoader(forget_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_eval_loader = DataLoader(retain_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(unseen_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 9. initialize architecture dynamically
    use_moe = getattr(args, 'use_moe', False)
    print(f"[*] initializing model: {args.model_name}{' (MoE)' if use_moe else ''}")
    if 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'deit' in args.model_name and use_moe:
        model = MoEDeiTArchitecture(
            model_name=args.model_name,
            num_classes=num_classes,
            num_experts=getattr(args, 'num_experts', 4),
            pretrained=args.pretrained,
            device=device,
        )
    elif 'deit' in args.model_name:
        model = DeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    else:
        raise ValueError(f"unsupported model prefix for {args.model_name}")

    # 10. setup optimizer and loss
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 11. training loop
    print("[*] starting training phase...")
    total_train_time = 0.0

    # MoE auxiliary loss hyper-parameters (read from config; defaults are conservative)
    if use_moe:
        lambda_sp  = getattr(args, 'lambda_sp',  0.1)
        lambda_bal = getattr(args, 'lambda_bal',  0.1)
        lambda_div = getattr(args, 'lambda_div',  0.01)
        ema_alpha  = getattr(args, 'ema_alpha',   0.99)
        M = model.num_experts
        ema_pi = torch.full((M,), 1.0 / M, device=device)  # uniform init
        print(f"[*] moe auxiliary losses: λ_sp={lambda_sp} λ_bal={lambda_bal} "
              f"λ_div={lambda_div} ema_α={ema_alpha}")

    for epoch in range(args.epochs):
        # -- train one epoch --
        model.train()
        total_loss = 0.0

        # start timer for pure training operations
        epoch_start_time = time.time()

        for batch in train_loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()
            logits, _ = model.forward_with_grad(images)

            l_task = criteria(logits, labels)

            if use_moe:
                # adapter.last_routing_weights and last_expert_outputs are
                # populated by model.forward_with_grad() above
                l_sp  = _compute_sp_loss(model.moe_adapters)
                l_bal, ema_pi = _compute_bal_loss(model.moe_adapters, ema_pi, ema_alpha)
                l_div = _compute_div_loss(model.moe_adapters)
                loss = (l_task
                        + lambda_sp  * l_sp
                        + lambda_bal * l_bal
                        + lambda_div * l_div)
            else:
                loss = l_task

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # stop timer before evaluation
        epoch_train_time = time.time() - epoch_start_time
        total_train_time += epoch_train_time
            
        avg_loss = total_loss / len(train_loader)
        
        # -- evaluate all metrics --
        fa_score = forget_acc(model, forget_eval_loader, device)
        ra_score = retain_acc(model, retain_eval_loader, device)
        ta_score = test_acc(model, test_loader, device)
        mia_score = mia(model, forget_eval_loader, unseen_loader, device)
        
        print(f"epoch [{epoch+1}/{args.epochs}] | loss: {avg_loss:.4f} | "
              f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
              f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })

    # 12. calculate final metrics (memory and total time)
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_memory_gb = 0.0
        
    print("\n[*] --- final training summary ---")
    print(f"[*] total training time (excluding evaluation): {total_train_time:.2f} seconds")
    print(f"[*] peak gpu memory usage: {peak_memory_gb:.4f} gb")
    
    if use_wandb:
        wandb.log({
            "total_train_time_sec": total_train_time,
            "peak_memory_gb": peak_memory_gb
        })

    # 13. save the final pretrained model
    save_path = os.path.join(args.output_dir, f"{yaml_filename}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[*] training complete. model saved to {save_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()