import torch
import torch.nn as nn
import timm
from architecture.based_model import BaseArchitecture
from architecture.module import MoEAdapter


class MoEDeiTArchitecture(BaseArchitecture):
    """
    DeiT with a MoEAdapter injected into every transformer block.

    The standard Mlp (FFN) inside each block is replaced in-place by a
    MoEAdapter. The DeiT backbone (attention layers, norms, patch embedding,
    cls/pos tokens) is frozen during both training and unlearning. Only the
    MoE adapter parameters and the classification head are trainable.

    Per-block structure (repeated ×N):
        Norm → Multi-head Attention  [frozen ❄️]
        Norm → MoEAdapter            [trainable 🔥]

    Public API (identical to BaseArchitecture for drop-in compatibility):
        forward_with_grad(x) → (logits, features)   train mode, grad ON
        inference(x)         → (logits, features)   eval mode,  no_grad

    MoE-specific API (used by ModularUnlearning):
        _forward(x)              → (logits, pi, h)
            pi: (B, M) routing weights averaged over all blocks and tokens
            h:  (B, D) CLS-token features
        self.moe_adapters        — list[MoEAdapter], one per transformer block
        self.moe_adapters[i].experts[m] — ExpertFFN for block i, expert m
        self.num_experts         — M
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
                f"unsupported model '{model_name}'. choose from {self.SUPPORTED_MODELS}"
            )

        featurizer = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = featurizer.num_features

        # --- replace block.mlp with MoEAdapter in every transformer block ---
        for block in featurizer.blocks:
            orig = block.mlp
            hidden_dim = orig.fc1.out_features
            drop = orig.drop1.p if hasattr(orig, 'drop1') else 0.0
            block.mlp = MoEAdapter(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                drop=drop,
            )

        # --- freeze backbone; keep only MoE adapter params trainable ---
        for name, param in featurizer.named_parameters():
            param.requires_grad_('mlp' in name)

        classifier_head = nn.Linear(embed_dim, num_classes)

        super().__init__(featurizer=featurizer, classifier_head=classifier_head, device=device)

        # direct handle to all adapters (same objects already inside featurizer.blocks)
        self.moe_adapters: list[MoEAdapter] = [
            block.mlp for block in self.featurizer.blocks
        ]
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_experts = num_experts

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def _forward(self, x: torch.Tensor):
        """
        Raw forward — exposes per-expert routing weights for scoring.

        Routing weights are averaged over all transformer blocks and all token
        positions to produce a single (B, M) summary per sample.

        Returns:
            logits: (B, num_classes)
            pi:     (B, num_experts)  mean routing weight per expert
            h:      (B, embed_dim)   CLS-token features
        """
        x = x.to(self.device)
        h = self.featurizer(x)               # runs all blocks; populates last_routing_weights
        logits = self.classifier_head(h)     # (B, C)

        # each adapter.last_routing_weights: (B, T, M)
        # stack → (num_blocks, B, T, M) → mean over blocks + tokens → (B, M)
        block_pis = [
            a.last_routing_weights
            for a in self.moe_adapters
            if a.last_routing_weights is not None
        ]
        if block_pis:
            # detach: _forward's pi is for external scoring only; training
            # losses use adapter.last_routing_weights directly (with grad).
            pi = torch.stack(block_pis, dim=0).mean(dim=(0, 2)).detach()  # (B, M)
        else:
            pi = torch.zeros(x.shape[0], self.num_experts, device=self.device)

        return logits, pi, h

    def forward(self, x: torch.Tensor):
        """Returns (logits, features) matching BaseArchitecture contract."""
        logits, _pi, h = self._forward(x)
        return logits, h

    def forward_with_grad(self, x: torch.Tensor):
        self.train()
        return self.forward(x)

    def inference(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
