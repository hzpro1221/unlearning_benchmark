import torch
import torch.nn as nn
import torchvision.models as models
from architecture.based_model import BaseArchitecture
class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim, num_classes, alpha_ratio=0.1):
        super(LearnableMaskLayer, self).__init__()
        self.mask = nn.Parameter(torch.rand(feature_dim, num_classes))
        self.alpha = torch.numel(self.mask) * alpha_ratio

    def get_channel_mask(self):
        return self.mask

    def get_density(self):
        return torch.norm(self.mask, p=1) / torch.numel(self.mask)

    def get_CSI(self):
        csi = 0
        mask_t = self.mask.transpose(0, 1)
        for idx in range(mask_t.size(0)):
            x = mask_t[idx].view(1, -1)
            for idy in range(mask_t.size(0)):
                if idx != idy:
                    y = mask_t[idy].view(1, -1)
                    csi += torch.cosine_similarity(x, y, dim=-1)
        return csi

    def _icnn_mask(self, x, labels):
        if self.training and labels is not None:
            batch_mask = self.mask[:, labels].t().view(x.size(0), x.size(1), 1, 1)
            return x * batch_mask
        return x

    def loss_function(self):
        mask_flat = self.mask.view(self.mask.size(0), -1)
        inner_product = torch.triu(torch.mm(mask_flat.transpose(0, 1), mask_flat), diagonal=1).sum()
        
        g_l1_norm = torch.norm(self.mask, p=1)
        l1_reg = torch.relu(g_l1_norm - self.alpha)
        
        return l1_reg, inner_product

    def channel_express(self, x, target):
        if not isinstance(target, list):
            target = [target.item() if isinstance(target, torch.Tensor) else target]
            
        target_masks = self.mask[:, target]
        keep_channel = (target_masks <= 0).all(dim=1).float()
        
        return x * keep_channel.view(1, -1, 1, 1)

    def clip_lmask(self):
        with torch.no_grad():
            self.mask.data = torch.clamp(self.mask.data, min=0.0, max=1.0)
            
            max_indices = torch.argmax(self.mask.data, dim=0)
            self.mask.data[max_indices, torch.arange(self.mask.size(1))] = 1.0
            
    def get_bar_Gc(self, forget_classes):
        with torch.no_grad():
            if not isinstance(forget_classes, list):
                forget_classes = [forget_classes]
                
            target_G = self.mask[:, forget_classes]
            bar_G_cj = (target_G <= 0).float()
            
            num_targets = len(forget_classes)
            bar_Gc = (torch.sum(bar_G_cj, dim=1) == num_targets).float()
            
            return bar_Gc.view(1, -1)

class KTPResNetFeaturizer(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=10, alpha_ratio=0.1):
        super().__init__()
        
        if 'resnet18' in model_name:
            backbone = models.resnet18(pretrained=pretrained)
            self.embed_dim = 512
        elif 'resnet50' in model_name:
            backbone = models.resnet50(pretrained=pretrained)
            self.embed_dim = 2048
        else:
            raise ValueError(f"Unsupported KTP ResNet: {model_name}")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.lmask = LearnableMaskLayer(feature_dim=self.embed_dim, num_classes=num_classes, alpha_ratio=alpha_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_outputs = self.embed_dim

    def forward(self, x, labels=None, target=None):
        x = self.feature_extractor(x)

        l1_reg, inner_product = 0.0, 0.0
        
        if labels is not None:
            x = self.lmask._icnn_mask(x, labels)
            l1_reg, inner_product = self.lmask.loss_function()
        elif target is not None:
            x = self.lmask.channel_express(x, target)

        x = torch.flatten(self.avgpool(x), 1)
        
        return (x, l1_reg, inner_product) if labels is not None else x


class ERM_KTP_Resnet(BaseArchitecture):
    def __init__(self, model_name='resnet50', num_classes=10, pretrained=True, alpha_ratio=0.1, device="cuda"):
        featurizer = KTPResNetFeaturizer(model_name=model_name, pretrained=pretrained, num_classes=num_classes, alpha_ratio=alpha_ratio)
        classifier_head = nn.Linear(featurizer.n_outputs, num_classes)
        
        super().__init__(featurizer=featurizer, classifier_head=classifier_head, device=device)
        self.num_classes = num_classes

    def forward(self, x, labels=None, target=None):
        features = self.featurizer(x, labels=labels, target=target)
        
        if isinstance(features, tuple):
            feat, l1_reg, inner_product = features
            return self.classifier_head(feat), l1_reg, inner_product
            
        return self.classifier_head(features)
            
    def forward_with_grad(self, x, labels=None, target=None):
        features = self.featurizer(x, labels=labels, target=target)
        
        if isinstance(features, tuple):
            feat, l1_reg, inner_product = features
            return self.classifier_head(feat), feat, l1_reg, inner_product
            
        return self.classifier_head(features), features

    def clip_lmask(self):
        self.featurizer.lmask.clip_lmask()
    
    def inference(self, x):
        self.eval()
        with torch.no_grad():
            features = self.featurizer(x)
            logits = self.classifier_head(features)
        return logits, features