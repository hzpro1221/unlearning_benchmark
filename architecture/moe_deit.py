import torch
import timm
import torch.nn as nn
from architecture.based_model import BaseArchitecture
from architecture.module import ExplicitMoEHead


class MoEDeiTArchitecture(BaseArchitecture):
    """
    DeiT featurizer + ExplicitMoEHead classification head.

    The DeiT backbone extracts CLS-token features z ∈ R^D.
    The MoE head blends M expert projections via soft routing, then classifies.

    Public API (identical to BaseArchitecture for drop-in compatibility):
        forward_with_grad(x) → (logits, h)   train mode, grad ON
        inference(x)         → (logits, h)   eval mode,  no_grad

    MoE-specific API (used by ModularUnlearning):
        _forward(x)             → (logits, pi, h)  raw output with routing weights
        self.moe_head           — ExplicitMoEHead module
        self.moe_head.experts   — nn.ModuleList[M] of expert Linear layers
        self.moe_head.router    — gating network G
        self.moe_head.classifier — classification head C
    """

    SUPPORTED_MODELS = [
        'deit_tiny_patch16_224',
        'deit_small_patch16_224',
        'deit_base_patch16_224',
        'deit_tiny_distilled_patch16_224',
        'deit_small_distilled_patch16_224',
        'deit_base_distilled_patch16_224',
    ]

    def __init__(
        self,
        model_name: str = 'deit_small_patch16_224',
        num_classes: int = 7,
        num_experts: int = 4,
        pretrained: bool = False,
        device: str = "cuda",
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported model '{model_name}'. choose from: {self.SUPPORTED_MODELS}"
            )

        featurizer = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = featurizer.num_features
        moe_head = ExplicitMoEHead(
            embed_dim=embed_dim,
            num_experts=num_experts,
            num_classes=num_classes,
        )

        # BaseArchitecture registers featurizer and classifier_head as submodules
        super().__init__(featurizer=featurizer, classifier_head=moe_head, device=device)

        # explicit alias so callers can write model.moe_head instead of model.classifier_head
        self.moe_head: ExplicitMoEHead = self.classifier_head
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_experts = num_experts

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def _forward(self, x: torch.Tensor):
        """
        Raw MoE computation — exposes routing weights for expert scoring.
        Moves x to device; does NOT manage train/eval or gradient context.

        Returns:
            logits: (B, num_classes)
            pi:     (B, num_experts) soft routing weights
            h:      (B, embed_dim)  blended expert features
        """
        x = x.to(self.device)
        z = self.featurizer(x)
        logits, pi, h = self.moe_head(z)
        return logits, pi, h

    def forward(self, x: torch.Tensor):
        """Returns (logits, features) matching BaseArchitecture contract."""
        logits, _pi, h = self._forward(x)
        return logits, h

    def forward_with_grad(self, x: torch.Tensor):
        """Train mode, gradients ON. Returns (logits, features)."""
        self.train()
        return self.forward(x)

    def inference(self, x: torch.Tensor):
        """Eval mode, no_grad. Returns (logits, features)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
