# main script for the unlearning pipeline
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
from architecture.erm_ktp_resnet import ERM_KTP_Resnet 
from architecture.asu_deity import ASUDeiTArchitecture  

from approx_algo.gradient_ascent import Gradient_Ascent
from approx_algo.l1_sparse import L1_Sparse
from approx_algo.random_labeling import Random_Labeling
from approx_algo.boundary_shrink import Boundary_Shrink
from approx_algo.finetune import Finetune
from approx_algo.module import Module
from approx_algo.scrub import SCRUB  
from approx_algo.sg_unlearn import SG_Unlearning 
from approx_algo.boundary_expanding import Boundary_Expanding 
from approx_algo.erm_ktp import ERM_KTP 
from approx_algo.asu import ASU  
from approx_algo.salun import SalUn 

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


def get_domain(dataset, idx):
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    data_tuple = dataset[idx]
    domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(domain_val)

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
    
    unlearn_algo = getattr(args, 'unlearn_algo', 'finetune').lower()
    unlearn_setting = getattr(args, 'unlearn_setting', 'random')
    project_name = f"{unlearn_setting}_unlearn"

    if unlearn_algo in ['erm_ktp', 'ktp'] and unlearn_setting != 'class':
        raise ValueError(
            f"\n[ERROR] Mathematical Constraint Violated:\n"
            f"ERM-KTP relies on severing class-specific feature channels.\n"
            f"It is strictly applicable only for `unlearn_setting: 'class'`.\n"
            f"Your current config uses: '{unlearn_setting}'."
        )
    
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

    forget_train_loader = DataLoader(ApplyTransform(forget_subset, get_forget_train_transform()), batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_train_loader = DataLoader(ApplyTransform(retain_subset, get_retain_train_transform()), batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_test_loader = DataLoader(ApplyTransform(forget_subset, get_forget_test_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_test_loader = DataLoader(ApplyTransform(retain_subset, get_retain_test_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(ApplyTransform(test_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(ApplyTransform(unseen_subset, get_unseen_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("\n" + "="*40)
    print(f"[*] loading model: {args.model_name}")
    
    if 'erm_ktp_resnet' in args.model_name:
        backbone_name = args.model_name.replace('erm_ktp_', '')
        model = ERM_KTP_Resnet(
            model_name=backbone_name, 
            num_classes=num_classes, 
            pretrained=False, 
            alpha_ratio=getattr(args, 'alpha_ratio', 0.1),
            device=device
        )
    elif 'asu_deit' in args.model_name:
        model = ASUDeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=False, device=device)
    elif 'resnet' in args.model_name:
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

    print("\n" + "="*40)
    print(f"[*] starting unlearning phase ({unlearn_algo})")

    algo_kwargs = {
        "model": model,
        "train_loader": None, 
        "test_loader": test_loader,
        "unseen_loader": unseen_loader,
        "forget_loader": forget_train_loader,
        "forget_test_loader": forget_test_loader,
        "retain_loader": retain_train_loader,
        "retain_test_loader": retain_test_loader,
        "optimizer": optimizer,
        "criteria": criteria,
        "num_epoch": args.epochs,
        "device": device
    }

    # OOP routing
    if unlearn_algo in ['gradient_ascent', 'ga']:
        algo_wrapper = Gradient_Ascent(**algo_kwargs)
    elif unlearn_algo == 'l1_sparse':
        algo_wrapper = L1_Sparse(**algo_kwargs, alpha=getattr(args, 'alpha', 0.1))
    elif unlearn_algo in ['random_labeling', 'rl']:
        algo_wrapper = Random_Labeling(**algo_kwargs)
    elif unlearn_algo == 'boundary_shrink':
        algo_wrapper = Boundary_Shrink(**algo_kwargs, epsilon=getattr(args, 'epsilon', 0.1))
    elif unlearn_algo in ['boundary_expanding', 'be']: 
        algo_wrapper = Boundary_Expanding(**algo_kwargs)
    elif unlearn_algo == 'scrub':  
        algo_wrapper = SCRUB(**algo_kwargs, alpha=getattr(args, 'alpha', 0.1))
    elif unlearn_algo in ['sg_unlearn', 'sg_unlearning']:
        algo_wrapper = SG_Unlearning(**algo_kwargs, alpha=getattr(args, 'alpha', 0.1)) 
    elif unlearn_algo in ['erm_ktp', 'ktp']: 
        algo_wrapper = ERM_KTP(
            **algo_kwargs,
            forget_classes=getattr(args, 'forget_classes', [0]),
            lambda_1=getattr(args, 'lambda_1', 1.0),
            lambda_2=getattr(args, 'lambda_2', 1.0),
            lambda_3=getattr(args, 'lambda_3', 0.1),
            warmup_epochs=getattr(args, 'warmup_epochs', 0),
            mask_period=getattr(args, 'mask_period', 3),
            mask_epoch_min=getattr(args, 'mask_epoch_min', 2)
        )
    elif unlearn_algo == 'asu':
        algo_wrapper = ASU(
            **algo_kwargs,
            tau=getattr(args, 'tau', 5.0),
            support_weight=getattr(args, 'support_weight', 1.0)
        )
    elif unlearn_algo == 'salun':  
        algo_wrapper = SalUn(
            **algo_kwargs,
            alpha=getattr(args, 'alpha', 1.0) 
        )
    elif unlearn_algo in ['module', 'module_unlearn_algo']:
        algo_wrapper = Module(
            **algo_kwargs,
            lambda_sparse=getattr(args, 'lambda_sparse', 1.0),
            lambda_balance=getattr(args, 'lambda_balance', 1.0),
            lambda_div=getattr(args, 'lambda_div', 1.0),
            alpha=getattr(args, 'alpha', 1.0),
            beta=getattr(args, 'beta', 1.0),
            gamma=getattr(args, 'gamma', 1.0),
            eta=getattr(args, 'eta', 1.0),
            k_u=getattr(args, 'k_u', 2)
        )
    elif unlearn_algo == 'finetune':
        algo_wrapper = Finetune(**algo_kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {unlearn_algo}")

    ckpt_prefix = os.path.join(args.output_dir, f"unlearned_{unlearn_algo}_{yaml_filename}")
    fa_threshold = getattr(args, 'fa_threshold', 0.8)

    total_unlearn_time = algo_wrapper.unlearn(fa_threshold=fa_threshold, ckpt_path=ckpt_prefix)

    print(f"\n[*] unlearning complete. Total time: {total_unlearn_time:.2f}s")
    print(f"[*] final model saved to {ckpt_prefix}.pt")
    
    wandb.finish()

if __name__ == "__main__":
    main()