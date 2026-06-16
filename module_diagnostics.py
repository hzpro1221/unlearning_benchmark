import os
import argparse
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from dataset.pytorch_dataset.cifar100 import CIFAR100Dataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.tiny_imagenet import TinyImageNetDataset

from dataset.transform.test_transform import get_test_transform
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform

from architecture.deity import DeiTArchitecture
from architecture.resnet import ResNetArchitecture
from architecture.module import ModuleArchitecture
from architecture.erm_ktp_resnet import ERM_KTP_Resnet
from architecture.asu_deity import ASUDeiTArchitecture 

class ApplyTransform(torch.utils.data.Dataset):
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

# safely get domain from dataset.
def get_domain(dataset, idx):
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    data_tuple = dataset[idx]
    domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(domain_val)

@torch.no_grad()
# funtion for evaluate acc.
def evaluate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits, _ = model.inference(images)
        
        predictions = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    
    return 100 * correct / total

def extract_routing_info(model, dataloader, device, num_experts, gate_k):
    all_probs = []

    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    if len(moe_layers) == 0:
        raise ValueError("[!] No DeepMoELayer found in the model!")

    model.eval() 
    with torch.no_grad(): 
        for batch in dataloader:
            images = batch[0].to(device)
            B = images.size(0) 
            
            _ = model.inference(images) 
            
            batch_pi_layers = []
            
            for moe_layer in moe_layers:
                pi_flat = moe_layer.last_pi_all 
                S = pi_flat.size(0) // B 
                # (B*S, E) -> (B, S, E) -> (B, E)
                pi_layer = pi_flat.view(B, S, num_experts).mean(dim=1)
                batch_pi_layers.append(pi_layer)
                
            # collect all layers -> (L, B, num_experts).
            batch_pi_stacked = torch.stack(batch_pi_layers, dim=0)
            all_probs.append(batch_pi_stacked.cpu())
            
    all_probs_cat = torch.cat(all_probs, dim=1)
    _, all_topk_indices = torch.topk(all_probs_cat, k=gate_k, dim=-1)

    return all_probs_cat, all_topk_indices

def compute_localization_diagnostics(model, forget_loader, retain_loader, device, num_experts, gate_k, k_u=2, alpha=1.0):
    print(f"[*] Extracting routing info... (Using alpha={alpha}, k_u={k_u})")
    
    pi_f, E_f = extract_routing_info(model, forget_loader, device, num_experts, gate_k)
    pi_r, E_r = extract_routing_info(model, retain_loader, device, num_experts, gate_k)

    L = pi_f.size(0) # number of MoE layers.
    N_f = pi_f.size(1)
    N_r = pi_r.size(1)

    total_FEM, total_ARR, total_RFO = 0.0, 0.0, 0.0

    # caculate metric for each layer.
    for l in range(L):
        pi_f_l, E_f_l = pi_f[l], E_f[l]
        pi_r_l, E_r_l = pi_r[l], E_r[l]

        mass_f = pi_f_l.mean(dim=0)
        mass_r = pi_r_l.mean(dim=0)

        # scoring k_u by difference.
        diff = mass_f - (alpha * mass_r)
        _, M_f = torch.topk(diff, k=k_u, dim=-1)

        # forget-expert routing mass (FEM).
        FEM_l = pi_f_l[:, M_f].sum(dim=1).mean().item()
        total_FEM += FEM_l

        # at-risk retain ratio (ARR).
        overlap_mask = torch.isin(E_r_l, M_f).any(dim=1)
        ARR_l = overlap_mask.float().mean().item()
        total_ARR += ARR_l

        # retain-forget routing overlap (RFO).
        E_f_hot = torch.zeros((N_f, num_experts), dtype=torch.float)
        E_r_hot = torch.zeros((N_r, num_experts), dtype=torch.float)
        
        E_f_hot.scatter_(1, E_f_l, 1.0)
        E_r_hot.scatter_(1, E_r_l, 1.0)

        intersection = E_f_hot @ E_r_hot.T 
        sum_f = E_f_hot.sum(dim=1, keepdim=True) 
        sum_r = E_r_hot.sum(dim=1, keepdim=True) 
        union = sum_f + sum_r.T - intersection
        
        iou_matrix = intersection / torch.clamp(union, min=1e-8)
        RFO_l = iou_matrix.mean().item()
        total_RFO += RFO_l

    # average all dataset.
    return total_FEM / L, total_ARR / L, total_RFO / L

def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config .yaml file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the .pth checkpoint file.")
    cmd_args = parser.parse_args()
    
    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    args = argparse.Namespace(**yaml_config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading dataset: {args.dataset}")
    
    # init dataset.
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

    # partition based on train/test/unseen splits.
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, _ = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

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

    train_loader = DataLoader(ApplyTransform(train_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ApplyTransform(final_test_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)
    forget_loader = DataLoader(ApplyTransform(forget_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)
    retain_loader = DataLoader(ApplyTransform(retain_subset, get_test_transform()), batch_size=args.batch_size, shuffle=False)

    print(f"[*] Initializing model: {args.model_name}")
    
    if 'module' in args.model_name:
        model = ModuleArchitecture(
            model_name=args.model_name, 
            num_classes=num_classes, 
            moe_layers=getattr(args, 'moe_layers', None),
            num_experts=args.num_experts,
            expert_depth=args.expert_depth,
            expert_hidden_ratio=args.expert_hidden_ratio,
            gate_k=args.gate_k,
            device=device
        )
    elif 'resnet' in args.model_name:
        model = ResNetArchitecture(model_name=args.model_name, num_classes=num_classes, device=device)
    else:
        raise NotImplementedError("Evaluation script is currently customized primarily for 'module' and 'resnet' architectures.")

    checkpoint = torch.load(cmd_args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print("\n" + "="*40)
    print("ACCURACY METRICS")
    print("="*40)
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%")

    if 'module' in args.model_name:
        print("\n" + "="*40)
        print("LOCALIZATION DIAGNOSTICS (MoDULE)")
        print("="*40)
        
        k_u = getattr(args, 'k_u', 2)
        alpha = getattr(args, 'alpha', 1.0)
        
        FEM, ARR, RFO = compute_localization_diagnostics(
            model=model, 
            forget_loader=forget_loader, 
            retain_loader=retain_loader, 
            device=device, 
            num_experts=args.num_experts,
            gate_k=args.gate_k,
            k_u=k_u,
            alpha=alpha
        )
        
        print(f"Forget-expert routing mass (FEM):    {FEM:.4f}")
        print(f"At-risk retain ratio (ARR):          {ARR:.4f}")
        print(f"Retain-forget routing overlap (RFO): {RFO:.4f}")

if __name__ == "__main__":
    main()