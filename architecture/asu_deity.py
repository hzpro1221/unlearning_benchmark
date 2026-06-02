import logging
import math
from collections import OrderedDict
from functools import partial
import sys
import os
from copy import deepcopy
from typing import Callable, Optional, Any, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from architecture.based_model import BaseArchitecture

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0, classifier=('head', 'head_dist')),
}

def update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter):
    """Updates default_cfg based on kwargs, and removes filtered kwargs."""
    for k in list(kwargs.keys()):
        if k in default_cfg:
            default_cfg[k] = kwargs[k]
        if kwargs_filter and k in kwargs_filter:
            kwargs.pop(k)

def _init_vit_weights(module: nn.Module, name: str = ''):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, 0.)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        pretrained_custom_load: bool = False,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Builds model with specified default_cfg and handles checkpoint loading. """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault('architecture', variant)

    if kwargs.pop('features_only', False):
        raise NotImplementedError('features_only not implemented for this ViT build.')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg

    num_classes_pretrained = getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        if pretrained_custom_load:
             _logger.warning("Custom load not supported in this simplified build, skipping.")
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict)

    return model

def load_pretrained(model, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
    state_dict = torch.hub.load_state_dict_from_url(
        model.default_cfg['url'], 
        map_location='cpu', 
        progress=True
    )
    
    if filter_fn is not None:
        state_dict = filter_fn(state_dict, model)
        
    if in_chans != 3 and 'patch_embed.proj.weight' in state_dict:
        state_dict['patch_embed.proj.weight'] = adapt_input_conv(
            in_chans, state_dict['patch_embed.proj.weight']
        )
        
    if num_classes == 0:
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            state_dict.pop(k, None)
        strict = False 
            
    model.load_state_dict(state_dict, strict=strict)

def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load=False,
        **kwargs)
    return model

def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    return _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)

def deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    return _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)

def deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)

def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    return _create_vision_transformer('deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)

def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    return _create_vision_transformer('deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)

def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer('deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)


def checkpoint_filter_fn(state_dict, model):
    # Ensures compatibility between legacy patch embedding weights and the current conv implementation
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
        
    return out_dict

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False, tau=1.0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Inject temperature tau into the attention logits
        attn = ((q @ k.transpose(-2, -1)) * self.scale) / tau
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, tau=1.0):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True, tau=tau)
            return attn
            
        x = x + self.drop_path(self.attn(self.norm1(x), tau=tau))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                act_layer=None, weight_init=''):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if distilled and num_classes > 0 else None

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, domain_index=None, tau=1.0):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            
        x = self.pos_drop(x + self.pos_embed)
        
        for blk in self.blocks:
            x = blk(x, tau=tau)
            
        x = self.norm(x)
        
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, domain_index=None, tau=1.0):
        x = self.forward_features(x, domain_index, tau=tau)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

class ASUDeiTFeaturizer(nn.Module):
    def __init__(self, model_name='deit_small_patch16_224', pretrained=True):
        super().__init__()
        if model_name not in default_cfgs:
            raise ValueError(f"Unknown model '{model_name}'. Supported: {sorted(default_cfgs.keys())}")

        if 'tiny' in model_name:
            depth, embed_dim = 12, 192
        elif 'small' in model_name:
            depth, embed_dim = 12, 384
        else:
            depth, embed_dim = 12, 768

        factory = globals()[model_name]

        self.vit = factory(
            pretrained=pretrained,
            num_classes=0,                  
            drop_path_rate=0.1
        )
        self.n_outputs = embed_dim
        self.model_name = model_name

    def forward(self, x, tau=1.0):
        out = self.vit.forward_features(x, tau=tau)
        return out[0] if isinstance(out, tuple) else out

class ASUDeiTArchitecture(BaseArchitecture):
    SUPPORTED_MODELS = [
        'asu_deit_tiny_patch16_224',
        'asu_deit_small_patch16_224',
        'asu_deit_base_patch16_224',
        'asu_deit_tiny_distilled_patch16_224',
        'asu_deit_small_distilled_patch16_224',
        'asu_deit_base_distilled_patch16_224'
    ]

    def __init__(self, model_name='asu_deit_small_patch16_224', num_classes=7, pretrained=True, device="cuda"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from: {self.SUPPORTED_MODELS}")

        backbone_name = model_name.replace('asu_', '', 1)

        featurizer = ASUDeiTFeaturizer(model_name=backbone_name, pretrained=pretrained)
        embed_dim = featurizer.n_outputs
        classifier_head = nn.Linear(embed_dim, num_classes)
        
        super().__init__(featurizer=featurizer, classifier_head=classifier_head, device=device)
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self, x, tau=1.0):
        features = self.featurizer(x, tau=tau)
        logits = self.classifier_head(features)
        return logits

    def forward_with_grad(self, x, tau=1.0):
        features = self.featurizer(x, tau=tau)
        logits = self.classifier_head(features)
        return logits, features
        
    def inference(self, x, tau=1.0):
        self.eval()
        with torch.no_grad():
            features = self.featurizer(x, tau=tau)
            logits = self.classifier_head(features)
        return logits, features