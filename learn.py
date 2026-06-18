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
from architecture.module import ModuleArchitecture
from architecture.erm_ktp_resnet import ERM_KTP_Resnet
from architecture.asu_deity import ASUDeiTArchitecture  

from approx_algo.gradient_ascent import Gradient_Ascent
from approx_algo.module import Module
from approx_algo.erm_ktp import ERM_KTP


class ApplyTransform(Dataset):
    """applies transforms to a subset, enforcing a base 224x224 resize for vit/resnet compatibility."""
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
    """safely extracts domain labels regardless of dataset tuple structure."""
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
    
    # handle for both Windows and Linux path.
    normalized_path = cmd_args.config.replace('\\', '/')
    
    path_no_ext = os.path.splitext(normalized_path)[0]

    if 'config/' in path_no_ext:
        rel_structure = 'config/' + path_no_ext.split('config/')[-1]
    else:
        rel_structure = os.path.basename(path_no_ext)
    rel_structure = os.path.normpath(rel_structure)

    if not hasattr(args, 'output_dir'):
        args.output_dir = os.path.join('checkpoint', rel_structure)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*40)
    print(f"[*] initializing learning pipeline")
    print(f"[*] config: {cmd_args.config}")
    print(f"[*] output_dir: {args.output_dir}")
    print(f"[*] device: {device}")

    unlearn_setting = getattr(args, 'unlearn_setting', 'random')

    if 'erm_ktp' in getattr(args, 'model_name', '').lower() and unlearn_setting != 'class':
        raise ValueError(
            f"\n[ERROR] Mathematical Constraint Violated:\n"
            f"ERM-KTP relies on severing class-specific feature channels.\n"
            f"It is strictly applicable only for `unlearn_setting: 'class'`.\n"
            f"Your current config uses: '{unlearn_setting}'."
        )
    
    wandb.login(key="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")
    wandb.init(
        project='learn',
        name=f"base_{yaml_filename}",
        config=yaml_config, 
        settings=wandb.Settings(start_method='thread')
    )

    # load dataset
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

    train_loader = DataLoader(ApplyTransform(train_subset, get_train_transform()), batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_train_loader = DataLoader(ApplyTransform(forget_subset, get_train_transform()), batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_train_loader = DataLoader(ApplyTransform(retain_subset, get_train_transform()), batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    forget_test_loader = DataLoader(ApplyTransform(forget_subset, get_forget_test_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_test_loader = DataLoader(ApplyTransform(retain_subset, get_retain_test_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(ApplyTransform(final_test_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(ApplyTransform(unseen_subset, get_unseen_transform()), batch_size=args.batch_size, shuffle=False, num_workers=4)

    if 'erm_ktp_resnet' in args.model_name:
        backbone_name = args.model_name.replace('erm_ktp_', '')
        model = ERM_KTP_Resnet(
            model_name=backbone_name, 
            num_classes=num_classes, 
            pretrained=args.pretrained,
            alpha_ratio=getattr(args, 'alpha_ratio', 0.1),
            device=device
        )
    elif 'asu_deit' in args.model_name:
        model = ASUDeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'deit' in args.model_name:
        model = DeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'module' in args.model_name:
        model = ModuleArchitecture(
            model_name=args.model_name, 
            num_classes=num_classes, 
            pretrained=args.pretrained,
            moe_layers=getattr(args, 'moe_layers', None),
            num_experts=args.num_experts,
            expert_depth=args.expert_depth,
            expert_hidden_ratio=args.expert_hidden_ratio,
            gate_k=args.gate_k,
            device=device
        )
        model._set_grad_mode("learning")
    else:
        raise ValueError(f"Unsupported model prefix for {args.model_name}")

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    algo_kwargs = {
        "model": model,
        "train_loader": train_loader,
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

    ckpt_prefix = os.path.join(args.output_dir, yaml_filename)
    print(f"\n[*] Starting Base Training Phase")

    if 'module' in args.model_name:
        algo_wrapper = Module(
            **algo_kwargs,
            lambda_sparse=getattr(args, 'lambda_sparse', 1.0),
            lambda_balance=getattr(args, 'lambda_balance', 1.0),
            lambda_div=getattr(args, 'lambda_div', 1.0),
            alpha=1.0, 
            beta=1.0, 
            gamma=1.0, 
            eta=1.0, 
            k_u=2 
        )
        ema_alpha = getattr(args, 'ema_alpha', 0.9)
        algo_wrapper.learn(ckpt_path=ckpt_prefix, ema_alpha=ema_alpha)
    elif 'erm_ktp' in args.model_name:
        algo_wrapper = ERM_KTP(
            **algo_kwargs,
            forget_classes=getattr(args, 'forget_classes', [0]),
            lambda_1=getattr(args, 'lambda_1', 1.0),
            lambda_2=getattr(args, 'lambda_2', 1.0),
            lambda_3=getattr(args, 'lambda_3', 0.1),
            warmup_epochs=getattr(args, 'warmup_epochs', 1),
            mask_period=getattr(args, 'mask_period', 2),
            mask_epoch_min=getattr(args, 'mask_epoch_min', 1)
        )
        algo_wrapper.learn(ckpt_path=ckpt_prefix)
    else:
        algo_wrapper = Gradient_Ascent(**algo_kwargs)
        algo_wrapper.learn(ckpt_path=ckpt_prefix)

    wandb.finish()

if __name__ == "__main__":
    main()