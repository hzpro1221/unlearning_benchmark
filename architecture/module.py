import logging
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

from architecture.based_model import BaseArchitecture

_logger = logging.getLogger(__name__)

# ==============================================================================
# MoE Components
# ==============================================================================

class DeepExpert(nn.Module):
    """N-layer MLP expert."""
    def __init__(self, model_dim, hidden_size, depth=2):
        super().__init__()
        dims = [model_dim] + [hidden_size] * (depth - 1) + [model_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        
        # Identity init for intermediate layers to start as pass-through
        for layer in self.layers[1:-1]:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.gelu(layer(x))
        return self.layers[-1](x)

class DeepMoELayer(nn.Module):
    """Top-K routing layer with deep experts."""
    def __init__(self, model_dim, hidden_size, num_experts=6, gate_k=1, expert_depth=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate_k = gate_k
        self.expert_depth = expert_depth
        self.experts = nn.ModuleList([DeepExpert(model_dim, hidden_size, expert_depth) for _ in range(num_experts)])
        self.router = nn.Linear(model_dim, num_experts, bias=False)

        self.last_pi_all = None
        self.last_pi = None
        self.last_h = None

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        
        logits = self.router(x_flat)

        self.last_pi_all = F.softmax(logits, dim=-1)

        topk_vals, topk_indices = logits.topk(self.gate_k, dim=-1)
        gate_weights = F.softmax(topk_vals, dim=-1)
        
        out = torch.zeros_like(x_flat)
        
        # Shape: (B*S, num_experts, D); E_raw represents H_m(x)
        E_raw = torch.zeros(x_flat.size(0), self.num_experts, D, device=x.device, dtype=x.dtype)
        
        for i in range(self.gate_k):
            indices = topk_indices[:, i]
            weights = gate_weights[:, i].unsqueeze(-1)
            
            for expert_idx, expert in enumerate(self.experts):
                mask = (indices == expert_idx)
                if mask.any():
                    raw_expert_out = expert(x_flat[mask])
                    
                    E_raw[mask, expert_idx, :] = raw_expert_out
                    out[mask] += raw_expert_out * weights[mask]
        
        self.last_pi = gate_weights 
        self.last_h = E_raw 

        return out.reshape(B, S, D)
# ==============================================================================
# Architecture Components
# ==============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_moe=True, 
                 num_experts=6, expert_depth=2, expert_hidden_ratio=2.0, gate_k=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
        expert_hidden_size = int(dim * expert_hidden_ratio)
        self.moe = DeepMoELayer(dim, expert_hidden_size, num_experts, gate_k, expert_depth) if use_moe else None

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.moe is not None:
            x = x + self.drop_path(self.moe(x))
        return x

class ModuleBackbone(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, use_moe=True, distilled=False,
                 num_experts=6, expert_depth=2, expert_hidden_ratio=2.0, gate_k=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                  norm_layer=norm_layer, act_layer=act_layer, use_moe=use_moe,
                  num_experts=num_experts, expert_depth=expert_depth, 
                  expert_hidden_ratio=expert_hidden_ratio, gate_k=gate_k)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
            
        return self.norm(x)[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))

# ==============================================================================
# Featurizer & Architecture Wrappers
# ==============================================================================

class ModuleFeaturizer(nn.Module):
    def __init__(self, model_name='deit_small_patch16_224', pretrained=True, use_moe=True,
                 num_experts=6, expert_depth=2, expert_hidden_ratio=2.0, gate_k=1):
        super().__init__()
        
        base_name = model_name.replace('_distilled', '')
        is_distilled = 'distilled' in model_name
        
        configs = {
            'deit_tiny_patch16_224': {'dim': 192, 'depth': 12, 'heads': 3},
            'deit_small_patch16_224': {'dim': 384, 'depth': 12, 'heads': 6},
            'deit_base_patch16_224': {'dim': 768, 'depth': 12, 'heads': 12}
        }
        
        if base_name not in configs:
            raise ValueError(f"Unknown model architecture '{model_name}'.")
            
        cfg = configs[base_name]
        
        self.model = ModuleBackbone(
            embed_dim=cfg['dim'], depth=cfg['depth'], num_heads=cfg['heads'], 
            use_moe=use_moe, distilled=is_distilled,
            num_experts=num_experts, expert_depth=expert_depth, 
            expert_hidden_ratio=expert_hidden_ratio, gate_k=gate_k
        )
        
        if pretrained:
            url_hash = 'a1311bcf' if 'tiny' in model_name else 'cd65a155' if 'small' in model_name else 'b5f2ef4d'
            url = f"https://dl.fbaipublicfiles.com/deit/{base_name}-{url_hash}.pth"
            
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove classifer heads to avoid shape mismatch
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                state_dict.pop(k, None)
            
            # strict=False allows ignoring randomly initialized MoE layers
            self.model.load_state_dict(state_dict, strict=False)
            
        self.n_outputs = cfg['dim']

    def forward(self, x):
        return self.model.forward_features(x)

class ModuleArchitecture(BaseArchitecture):
    SUPPORTED_MODELS = [
        'module_tiny_patch16_224',
        'module_small_patch16_224',
        'module_base_patch16_224',
        'module_tiny_distilled_patch16_224',
        'module_small_distilled_patch16_224',
        'module_base_distilled_patch16_224'
    ]

    def __init__(self, model_name='module_small_patch16_224', num_classes=7, pretrained=True, device="cuda",
                 num_experts=6, expert_depth=2, expert_hidden_ratio=2.0, gate_k=1):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' is not supported.")

        # Map 'module_' prefix back to 'deit_' for fetching pretrained weights
        deit_model_name = model_name.replace('module_', 'deit_')
        
        featurizer = ModuleFeaturizer(
            model_name=deit_model_name, pretrained=pretrained,
            num_experts=num_experts, expert_depth=expert_depth, 
            expert_hidden_ratio=expert_hidden_ratio, gate_k=gate_k
        )
        
        embed_dim = featurizer.n_outputs
        classifier_head = nn.Linear(embed_dim, num_classes)
        
        super().__init__(featurizer=featurizer, classifier_head=classifier_head, device=device)
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self._print_param_counts()

    def _set_grad_mode(self, mode="learning"):
        """Freeze/unfreeze specific parts of the network."""
        if mode == "learning":
            for param in self.parameters():
                param.requires_grad = True
                
        elif mode == "unlearning":
            # Freeze backbone and router, unfreeze only internal MoE experts
            for name, param in self.named_parameters():
                param.requires_grad = ("moe" in name and "router" not in name)

    def _count_params(self, module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _print_param_counts(self):
        import timm 

        feat_t, feat_tr = self._count_params(self.featurizer)
        head_t, head_tr = self._count_params(self.classifier_head)

        moe_expert_t = moe_router_t = num_moe_layers = 0
        num_experts_per_layer = gate_k = expert_depth = 0
        expert_hidden_ratio = 0.0

        for name, module in self.featurizer.named_modules():
            if isinstance(module, DeepMoELayer):
                num_moe_layers += 1
                
                # Scrape MoE configuration
                num_experts_per_layer = getattr(module, 'num_experts', 0)
                gate_k = getattr(module, 'gate_k', 0)
                expert_depth = getattr(module, 'expert_depth', 0)
                
                # Infer hidden ratio from the first linear layer of the first expert
                if len(module.experts) > 0 and len(module.experts[0].layers) > 0:
                    first_linear = module.experts[0].layers[0]
                    expert_hidden_ratio = first_linear.out_features / first_linear.in_features

                moe_expert_t += sum(p.numel() for p in module.experts.parameters())
                moe_router_t += sum(p.numel() for p in module.router.parameters())

        total_t = feat_t + head_t
        total_tr = feat_tr + head_tr

        # Compare with original DeiT
        deit_model_name = self.model_name.replace('module_', 'deit_')
        try:
            baseline_model = timm.create_model(deit_model_name, pretrained=False, num_classes=self.num_classes)
            baseline_t = sum(p.numel() for p in baseline_model.parameters())
            del baseline_model
        except Exception:
            baseline_t = 0

        def fmt(n): return f'{n:>13,}  ({n / 1e6:6.2f}M)'

        print(f'\n[{type(self).__name__}] param counts  —  backbone={self.model_name}')
        print(f'  MoE Layers           : {num_moe_layers}')
        print(f'  Experts per layer    : {num_experts_per_layer}')
        print(f'  Expert depth         : {expert_depth}')
        print(f'  Expert hidden ratio  : {expert_hidden_ratio:.1f}')
        print(f'  Router top-k (gate_k): {gate_k}')
        
        print(f'  featurizer (ViT+MoE): {fmt(feat_t)}  trainable={fmt(feat_tr)}')
        print(f'    ├─ MoE experts    : {fmt(moe_expert_t)}')
        print(f'    └─ MoE routers    : {fmt(moe_router_t)}')
        print(f'  classifier head     : {fmt(head_t)}  trainable={fmt(head_tr)}')
        print(f'  ─────────────────────────────────────────────')
        print(f'  TOTAL (with MoE)    : {fmt(total_t)}  trainable={fmt(total_tr)}')
        
        if baseline_t > 0:
            overhead_pct = ((total_t / baseline_t) - 1) * 100
            print(f'  Original DeiT Base  : {fmt(baseline_t)}')
            print(f'  MoE Overhead        : {fmt(total_t - baseline_t)} (+{overhead_pct:.1f}%)\n')
        else:
            print()