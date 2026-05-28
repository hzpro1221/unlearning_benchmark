import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms
import wandb
import yaml

from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset

from dataset.transform.train_transform import get_train_transform
from dataset.transform.test_transform import get_test_transform
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
from dataset.transform.unseen_transform import get_unseen_transform

from architecture.deity import DeiTArchitecture
from architecture.resnet import ResNetArchitecture
from architecture.module import ModuleArchitecture, DeepMoELayer

from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia


class ApplyTransform(Dataset):
    """Applies transforms to a subset, enforcing a base 224x224 resize for ViT/ResNet compatibility."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        data = self.subset[idx]
        image = self.resize(data[0])
        
        if self.transform:
            image = self.transform(image)
            
        return (image,) + data[1:]

    def __len__(self):
        return len(self.subset)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def get_domain(dataset, idx):
    """Safely extracts domain labels regardless of dataset tuple structure."""
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    
    data_tuple = dataset[idx]
    domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(domain_val)


def main():
    parser = argparse.ArgumentParser(description="Train a base model from yaml config.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config .yaml file.")
    cmd_args = parser.parse_args()
    
    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    args = argparse.Namespace(**yaml_config)
    yaml_filename = os.path.splitext(os.path.basename(cmd_args.config))[0]

    if not hasattr(args, 'output_dir'):
        args.output_dir = f'checkpoint/learn/{yaml_filename}'

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    wandb.login(key="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")
    wandb.init(
        project='learn',
        name=f"base_{yaml_filename}",
        config=yaml_config, 
        settings=wandb.Settings(start_method='thread')
    )

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
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Primary Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, unseen_subset = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    # Secondary Split (Unlearning Target Identification)
    unlearn_setting = getattr(args, 'unlearn_setting', 'random')
    
    if unlearn_setting == 'random':
        forget_size = int(getattr(args, 'forget_ratio', 0.1) * train_size)
        forget_subset, retain_subset = random_split(
            train_subset, [forget_size, train_size - forget_size], generator=generator
        )
        final_test_subset = test_subset
        
    elif unlearn_setting in ['class', 'domain']:
        target_list = getattr(args, f'forget_{unlearn_setting}es', [0])
        if not isinstance(target_list, list): 
            target_list = [target_list]
        
        f_tr_idx, r_tr_idx, r_te_idx = [], [], []
        
        for idx in train_subset.indices:
            val = full_dataset.labels[idx] if unlearn_setting == 'class' else get_domain(full_dataset, idx)
            (f_tr_idx if val in target_list else r_tr_idx).append(idx)
            
        for idx in test_subset.indices:
            val = full_dataset.labels[idx] if unlearn_setting == 'class' else get_domain(full_dataset, idx)
            if val not in target_list: 
                r_te_idx.append(idx)
                
        forget_subset = Subset(full_dataset, f_tr_idx)
        retain_subset = Subset(full_dataset, r_tr_idx)
        final_test_subset = Subset(full_dataset, r_te_idx)
    else:
        raise ValueError("unlearn_setting must be 'random', 'class', or 'domain'")

    train_set = ApplyTransform(train_subset, transform=get_train_transform())
    forget_eval_set = ApplyTransform(forget_subset, transform=get_forget_test_transform())
    retain_eval_set = ApplyTransform(retain_subset, transform=get_retain_test_transform())
    test_set = ApplyTransform(final_test_subset, transform=get_test_transform())
    unseen_set = ApplyTransform(unseen_subset, transform=get_unseen_transform())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_eval_loader = DataLoader(forget_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_eval_loader = DataLoader(retain_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(unseen_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'deit' in args.model_name:
        model = DeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'module' in args.model_name:
        model = ModuleArchitecture(
            model_name=args.model_name, 
            num_classes=num_classes, 
            pretrained=args.pretrained,
            num_experts=args.num_experts,
            expert_depth=args.expert_depth,
            expert_hidden_ratio=args.expert_hidden_ratio,
            gate_k=args.gate_k,
            device=device
            )
        model._set_grad_mode("learning")

        model = torch.compile(model)
    else:
        raise ValueError(f"Unsupported model prefix for {args.model_name}")

    if 'resnet' in args.model_name or 'deit' in args.model_name:
        criteria = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        total_train_time = 0.0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            
            epoch_start_time = time.time()
            
            for batch in train_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                
                optimizer.zero_grad()
                logits, _ = model.forward_with_grad(images)
                loss = criteria(logits, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time
                
            avg_loss = total_loss / len(train_loader)
            
            fa_score = forget_acc(model, forget_eval_loader, device)
            ra_score = retain_acc(model, retain_eval_loader, device)
            ta_score = test_acc(model, test_loader, device)
            mia_score = mia(model=model, unseen_loader=unseen_loader, forget_loader=forget_eval_loader, device=device)
            
            print(f"epoch [{epoch+1}/{args.epochs}] | loss: {avg_loss:.4f} | "
                f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{yaml_filename}_epoch_{epoch+1}.pt"))
    # the criteria for module is specific designed, different to which for resnet or deit
    elif 'module' in args.model_name:
        # --- Balance Loss EMA Setup ---
        # Activate EMA if the batch size is 8 or smaller
        use_ema = args.batch_size <= 8
        ema_states = {}
        ema_alpha = getattr(args, 'ema_alpha', 0.9)

        def loss_sparse(pi):
            entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
            return entropy.mean()

        def loss_balance(pi, module_name):
            M = pi.size(-1)
            mean_pi = pi.mean(dim=0) 
            
            if use_ema:
                if module_name not in ema_states:
                    ema_states[module_name] = torch.ones_like(mean_pi) / M
                effective_pi = ema_alpha * ema_states[module_name] + (1 - ema_alpha) * mean_pi
                
                ema_states[module_name] = effective_pi.detach()
            else:
                effective_pi = mean_pi

            return ((effective_pi - 1.0 / M) ** 2).sum()

        def loss_diversity(h_stack, eps=1e-6):
            B, M, r = h_stack.shape
            
            if M < 2: 
                return h_stack.new_zeros(())
            
            loss = h_stack.new_zeros(1).squeeze()
            
            H_tilde = []
            for m in range(M):
                H_m = h_stack[:, m, :]  # Shape: (B, r)

                norm_F = H_m.norm(p='fro').clamp(min=eps)
                
                # H_tilde_m = H_m / (||H_m||_F + eps)
                H_tilde_m = H_m / norm_F
                H_tilde.append(H_tilde_m)
            for m in range(M):
                for n in range(M):
                    if m == n: 
                        continue

                    # Cross-correlation matrix: (H_tilde_m^T @ H_tilde_n)
                    C = (H_tilde[m].T @ H_tilde[n])
                    loss += (C ** 2).sum()
            return loss
        
        # lambda for losses
        lambda_sparse = args.lambda_sparse
        lambda_balance = args.lambda_balance
        lambda_div = args.lambda_div

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criteria = nn.CrossEntropyLoss()
        total_train_time = 0.0

        for epoch in range(args.epochs):
            model.train()
            epoch_start_time = time.time() 
            
            running_total, running_ce, running_sp, running_bal, running_div = 0.0, 0.0, 0.0, 0.0, 0.0

            for batch in train_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                
                optimizer.zero_grad()
                logits, _ = model.forward_with_grad(images)
                
                all_pi, all_h, moe_names = [], [], []
                for name, module in model.featurizer.model.named_modules():
                    if isinstance(module, DeepMoELayer):
                        all_pi.append(module.last_pi)
                        all_h.append(module.last_h)
                        moe_names.append(name) 

                ce_loss = criteria(logits, labels)
                sp_loss = sum([loss_sparse(pi) for pi in all_pi]) / len(all_pi)
                bal_loss = sum([loss_balance(pi, n) for pi, n in zip(all_pi, moe_names)]) / len(all_pi)
                div_loss = sum([loss_diversity(h) for h in all_h]) / len(all_h)
                
                t_loss = ce_loss + (args.lambda_sparse * sp_loss) + (args.lambda_balance * bal_loss) + (args.lambda_div * div_loss)

                t_loss.backward()
                optimizer.step()
                
                running_total += t_loss.item()
                running_ce += ce_loss.item()
                running_sp += sp_loss.item()
                running_bal += bal_loss.item()
                running_div += div_loss.item()

            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time
            
            num_batches = len(train_loader)
            avg_loss = running_total / num_batches
            avg_ce = running_ce / num_batches 
            avg_sp = running_sp / num_batches 
            avg_bal = running_bal / num_batches 
            avg_div = running_div / num_batches 
            
            fa_score = forget_acc(model, forget_eval_loader, device)
            ra_score = retain_acc(model, retain_eval_loader, device)
            ta_score = test_acc(model, test_loader, device)
            mia_score = mia(model=model, unseen_loader=unseen_loader, forget_loader=forget_eval_loader, device=device)
            
            print(f"epoch [{epoch+1}/{args.epochs}] | "
              f"total_loss: {avg_loss:.4f} (ce: {avg_ce:.4f}, sp: {avg_sp:.4f}, bal: {avg_bal:.4f}, div: {avg_div:.4f}) | "
              f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
              f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "ce_loss": running_ce / num_batches,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{yaml_filename}_epoch_{epoch+1}.pt"))

    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        
    wandb.log({
        "total_train_time_sec": total_train_time,
        "peak_memory_gb": peak_memory_gb
    })

    save_path = os.path.join(args.output_dir, f"{yaml_filename}.pt")
    torch.save(model.state_dict(), save_path)
    wandb.finish()

if __name__ == "__main__":
    main()