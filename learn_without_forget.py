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

# import dataloaders
from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset

# import transforms
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
from dataset.transform.retain_train_transform import get_retain_train_transform
from dataset.transform.test_transform import get_test_transform
from dataset.transform.unseen_transform import get_unseen_transform

# import architectures
from architecture.deity import DeiTArchitecture
from architecture.resnet import ResNetArchitecture

# import metrics
from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia

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
    parser = argparse.ArgumentParser(description="train gold standard (retrain from scratch) using yaml config.")
    parser.add_argument('--config', type=str, required=True, help="path to the config .yaml file.")
    cmd_args = parser.parse_args()
    
    # 1. load yaml config
    print(f"[*] loading config from {cmd_args.config}")
    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    args = argparse.Namespace(**yaml_config)
    
    # default output directory for retrained models
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'checkpoint/learn_withou_forget'
        
    yaml_filename = os.path.splitext(os.path.basename(cmd_args.config))[0]

    # 2. setup environment
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] using device: {device}")
    
    # 3. initialize wandb
    print("[*] initializing wandb...")
    wandb.login(key="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")
    run_name = f"retrain_{yaml_filename}"
    
    wandb.init(
        project=getattr(args, 'wandb_project', 'retrain_gold_standard'),
        name=run_name,
        config=yaml_config, 
        settings=wandb.Settings(start_method='thread')
    )

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
            if lbl in forget_classes: f_tr_idx.append(idx)
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

    print(f"[*] split sizes -> retain (train): {len(retain_subset)} | forget (omitted): {len(forget_subset)}")
    print(f"[*] split sizes -> test: {len(final_test_subset)} | unseen: {unseen_size}")

    # 7. apply explicit transforms
    # **CRITICAL DIFFERENCE**: we ONLY augment the retain set for training.
    retain_train_set = ApplyTransform(retain_subset, transform=get_retain_train_transform())
    
    # evaluation sets strictly use test transforms (no randomness)
    forget_eval_set = ApplyTransform(forget_subset, transform=get_forget_test_transform())
    retain_eval_set = ApplyTransform(retain_subset, transform=get_retain_test_transform())
    test_set = ApplyTransform(final_test_subset, transform=get_test_transform())
    unseen_set = ApplyTransform(unseen_subset, transform=get_unseen_transform())

    # 8. create dataloaders
    retain_train_loader = DataLoader(retain_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    forget_eval_loader = DataLoader(forget_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_eval_loader = DataLoader(retain_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(unseen_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 9. initialize architecture dynamically (FROM SCRATCH)
    print(f"[*] initializing model: {args.model_name}")
    if 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    elif 'deit' in args.model_name:
        model = DeiTArchitecture(model_name=args.model_name, num_classes=num_classes, pretrained=args.pretrained, device=device)
    else:
        raise ValueError(f"unsupported model prefix for {args.model_name}")

    # 10. setup optimizer and loss
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 11. training loop (ONLY ON RETAIN SET)
    print("[*] starting retrain (gold standard) phase...")
    total_train_time = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # start timer for pure training operations
        epoch_start_time = time.time()
        
        # **CRITICAL DIFFERENCE**: iterating only over retain_train_loader
        for batch in retain_train_loader:
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
            
        avg_loss = total_loss / len(retain_train_loader)
        
        # -- evaluate all metrics to mirror unlearning scripts --
        fa_score = forget_acc(model, forget_eval_loader, device)
        ra_score = retain_acc(model, retain_eval_loader, device)
        ta_score = test_acc(model, test_loader, device)
        mia_score = mia(model, forget_eval_loader, unseen_loader, device)
        
        print(f"epoch [{epoch+1}/{args.epochs}] | loss: {avg_loss:.4f} | "
              f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
              f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s")
        
        # log to wandb
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
        
    print("\n[*] --- final retrain summary ---")
    print(f"[*] total training time (excluding evaluation): {total_train_time:.2f} seconds")
    print(f"[*] peak gpu memory usage: {peak_memory_gb:.4f} gb")
    
    wandb.log({
        "total_train_time_sec": total_train_time,
        "peak_memory_gb": peak_memory_gb
    })

    # 13. save the final retrained model
    save_path = os.path.join(args.output_dir, f"retrained_{yaml_filename}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[*] retraining complete. gold standard model saved to {save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()