# imports and dependencies
import os
import argparse
import random
import time
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms
import wandb
import yaml

from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset

from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.forget_train_transform import get_forget_train_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
from dataset.transform.retain_train_transform import get_retain_train_transform
from dataset.transform.test_transform import get_test_transform
from dataset.transform.unseen_transform import get_unseen_transform

from architecture.deity import DeiTArchitecture
from architecture.resnet import ResNetArchitecture
from architecture.module import ModuleArchitecture

from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia

from approx_algo.boundary_shrink import boundary_shrink
from approx_algo.gradient_ascent import gradient_ascent
from approx_algo.l1_sparse import l1_sparse
from approx_algo.random_labeling import random_labeling
from approx_algo.module_unlearn_algo import module_unlearn_algo

class ApplyTransform(Dataset):
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


def main():
    parser = argparse.ArgumentParser(description="Unlearn a subset using a yaml config.")
    parser.add_argument('--config', type=str, required=True, help="Path to config.")
    cmd_args = parser.parse_args()
    
    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    args = argparse.Namespace(**yaml_config)
    yaml_filename = os.path.splitext(os.path.basename(cmd_args.config))[0]

    if not hasattr(args, 'output_dir'):
        args.output_dir = f'checkpoint/unlearn/{yaml_filename}'

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*40)
    print(f"[*] initializing unlearning pipeline")
    print(f"[*] config: {cmd_args.config}")
    print(f"[*] device: {device}")
    
    unlearn_algo = getattr(args, 'unlearn_algo', 'finetune')
    unlearn_setting = getattr(args, 'unlearn_setting', 'random')
    project_name = f"{unlearn_setting}_unlearn"
    
    wandb.login(key="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")
    wandb.init(project=project_name, name=f"unlearn_{unlearn_algo}_{yaml_filename}", config=yaml_config)

    print("\n" + "="*40)
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
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, unseen_subset = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    print(f"[*] unlearning setting: {unlearn_setting}")
    
    if unlearn_setting == 'random':
        forget_ratio = getattr(args, 'forget_ratio', 0.1)
        forget_size = int(forget_ratio * train_size)
        forget_subset, retain_subset = random_split(
            train_subset, [forget_size, train_size - forget_size], generator=generator
        )
        print(f"[*] split sizes -> retain: {len(retain_subset)} | forget: {len(forget_subset)} | test: {len(test_subset)} | unseen: {len(unseen_subset)}")

    elif unlearn_setting == 'class':
        forget_classes = getattr(args, 'forget_classes', [0])
        if not isinstance(forget_classes, list):
            forget_classes = [forget_classes]
            
        print(f"[*] target classes to unlearn: {forget_classes}")
        forget_train_indices, retain_train_indices, retain_test_indices = [], [], []

        for idx in train_subset.indices:
            label = full_dataset.labels[idx]
            if label in forget_classes:
                forget_train_indices.append(idx)
            else:
                retain_train_indices.append(idx)
                
        for idx in test_subset.indices:
            label = full_dataset.labels[idx]
            if label not in forget_classes:
                retain_test_indices.append(idx)
                
        forget_subset = Subset(full_dataset, forget_train_indices)
        retain_subset = Subset(full_dataset, retain_train_indices)
        test_subset = Subset(full_dataset, retain_test_indices) 
        print(f"[*] split sizes -> retain: {len(retain_subset)} | forget: {len(forget_subset)} | test: {len(test_subset)} | unseen: {len(unseen_subset)}")

    elif unlearn_setting == 'domain':
        forget_domains = getattr(args, 'forget_domains', [0])
        if not isinstance(forget_domains, list):
            forget_domains = [forget_domains]
            
        print(f"[*] target domains to unlearn: {forget_domains}")
        forget_train_indices, retain_train_indices, retain_test_indices = [], [], []

        def get_domain(dataset, idx):
            if hasattr(dataset, 'domains'):
                return dataset.domains[idx]
            else:
                data_tuple = dataset[idx]
                domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
                return int(domain_val)

        for idx in train_subset.indices:
            domain = get_domain(full_dataset, idx)
            if domain in forget_domains:
                forget_train_indices.append(idx)
            else:
                retain_train_indices.append(idx)
                
        for idx in test_subset.indices:
            domain = get_domain(full_dataset, idx)
            if domain not in forget_domains:
                retain_test_indices.append(idx)
                
        forget_subset = Subset(full_dataset, forget_train_indices)
        retain_subset = Subset(full_dataset, retain_train_indices)
        test_subset = Subset(full_dataset, retain_test_indices) 
        print(f"[*] split sizes -> retain: {len(retain_subset)} | forget: {len(forget_subset)} | test: {len(test_subset)} | unseen: {len(unseen_subset)}")
        
    else:
        raise ValueError(f"Unsupported unlearning setting: {unlearn_setting}")

    forget_train_loader = DataLoader(ApplyTransform(forget_subset, get_forget_train_transform()), 
                                    batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_train_loader = DataLoader(ApplyTransform(retain_subset, get_retain_train_transform()), 
                                    batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_eval_loader = DataLoader(ApplyTransform(forget_subset, get_forget_test_transform()), 
                                   batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_eval_loader = DataLoader(ApplyTransform(retain_subset, get_retain_test_transform()), 
                                   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(ApplyTransform(test_subset, get_test_transform()), 
                            batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(ApplyTransform(unseen_subset, get_unseen_transform()), 
                              batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("\n" + "="*40)
    print(f"[*] loading model: {args.model_name}")
    if 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=False, device=device)
    elif 'deit' in args.model_name:
        model = DeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=False, device=device)
    elif 'module' in args.model_name:
        model = ModuleArchitecture(
            model_name=args.model_name, 
            num_classes=num_classes, 
            pretrained=False,
            num_experts=args.num_experts,
            expert_depth=args.expert_depth,
            expert_hidden_ratio=args.expert_hidden_ratio,
            gate_k=args.gate_k,
            device=device
        )
        model._set_grad_mode("unlearning")
        model = torch.compile(model)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    if not os.path.exists(args.pretrained_model_path):
        raise FileNotFoundError(f"Invalid path: {args.pretrained_model_path}")
    
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criteria = nn.CrossEntropyLoss()

    fa_threshold = getattr(args, 'fa_threshold', 0.8)
    total_unlearn_time = 0.0
    
    selected_experts_per_layer = []
    origin_model = None

    print("\n" + "="*40)
    print(f"[*] starting unlearning phase ({unlearn_algo})")

    if unlearn_algo == 'module_unlearn_algo':
        origin_model = copy.deepcopy(model)
        origin_model.eval()
        for param in origin_model.parameters():
            param.requires_grad = False

        def get_routing_mass(loader):
            model.eval()
            masses = None
            total_tokens = 0
            with torch.no_grad():
                for batch in loader:
                    images = batch[0].to(device)
                    model(images)
                    
                    batch_masses = []
                    num_tokens = 0
                    for _, m in model.named_modules():
                        if m.__class__.__name__ == 'DeepMoELayer':
                            batch_masses.append(m.last_pi_all.sum(dim=0))
                            num_tokens = m.last_pi_all.size(0)
                            
                    if masses is None:
                        masses = batch_masses
                    else:
                        masses = [m + b for m, b in zip(masses, batch_masses)]
                    total_tokens += num_tokens
            return [m / max(total_tokens, 1) for m in masses]

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        if unlearn_algo == 'module_unlearn_algo':
            selected_experts_per_layer = []
            forget_mass = get_routing_mass(forget_train_loader)
            retain_mass = get_routing_mass(retain_train_loader)
            
            moe_layers = [m for _, m in model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']
            model._set_grad_mode("unlearning")
            
            alpha = getattr(args, 'alpha', 1.0)
            k_u = getattr(args, 'k_u', 2)
            
            for l_idx, m in enumerate(moe_layers):
                rho_m = forget_mass[l_idx] - alpha * retain_mass[l_idx]
                _, topk_indices = rho_m.topk(k_u, dim=-1)
                selected_experts = topk_indices.tolist()
                selected_experts_per_layer.append(selected_experts)
                
                for expert_idx, expert in enumerate(m.experts):
                    is_selected = expert_idx in selected_experts
                    for param in expert.parameters():
                        param.requires_grad = is_selected

        if unlearn_algo == 'boundary_shrink':
            avg_loss = boundary_shrink(model, criteria, optimizer, forget_train_loader, device=device)
        elif unlearn_algo == 'ga':
            avg_loss = gradient_ascent(model, criteria, optimizer, forget_train_loader, device=device)
        elif unlearn_algo == 'l1_sparse':
            avg_loss = l1_sparse(model, criteria, optimizer, forget_train_loader, device=device)
        elif unlearn_algo == 'rl':
            avg_loss = random_labeling(model, criteria, optimizer, forget_train_loader, device=device)
        elif unlearn_algo == 'module_unlearn_algo':
            avg_loss = module_unlearn_algo(
                model=model, 
                origin_model=origin_model, 
                criteria=criteria, 
                optimizer=optimizer, 
                forget_loader=forget_train_loader, 
                retain_loader=retain_train_loader, 
                selected_experts_per_layer=selected_experts_per_layer,
                device=device,
                beta=getattr(args, 'beta', 1.0),
                gamma=getattr(args, 'gamma', 1.0),
                eta=getattr(args, 'eta', 1.0)
            )
        elif unlearn_algo == 'finetune':
            model.train()
            total_loss = 0.0
            for batch in retain_train_loader:
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                logits, _ = model.forward_with_grad(images)
                loss = criteria(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(retain_train_loader)
        else:
            raise ValueError(f"Unsupported algorithm: {unlearn_algo}")
            
        epoch_time = time.time() - epoch_start_time
        total_unlearn_time += epoch_time
        
        print(f"[*] evaluating epoch {epoch+1}...")
        fa_score = forget_acc(model, forget_eval_loader, device)
        ra_score = retain_acc(model, retain_eval_loader, device)
        ta_score = test_acc(model, test_loader, device)
        mia_score = mia(model, forget_eval_loader, unseen_loader, device)
        
        print(f"--> Epoch [{epoch+1}/{args.epochs}] | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f}")
        print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
        print("-" * 40)
        
        wandb.log({"epoch": epoch+1, "unlearn_loss": avg_loss, "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score})
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"{yaml_filename}_epoch_{epoch+1}.pt"))

        if fa_score <= fa_threshold:
            print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
            break

    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"\n[*] unlearning complete. Total time: {total_unlearn_time:.2f}s | Peak memory: {peak_memory_gb:.2f} GB")
    wandb.log({"total_unlearn_time": total_unlearn_time, "peak_memory_gb": peak_memory_gb})
    
    final_model_path = os.path.join(args.output_dir, f"unlearned_{unlearn_algo}_{yaml_filename}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"[*] final model saved to {final_model_path}")
    wandb.finish()

if __name__ == "__main__":
    main()