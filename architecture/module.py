import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitMoEHead(nn.Module):
    """
    Mixture-of-Experts classification head with explicit routing exposure.

    Mathematical flow:
        h_m = E_m(z)            individual expert projections   (B, D) × M
        π(z) = softmax(G(z))    soft routing weights             (B, M)
        h(z) = Σ_m π_m · h_m   expert-blended feature vector   (B, D)
        ŷ    = C(h(z))          class logits                    (B, num_classes)

    Attributes:
        experts    (nn.ModuleList): M expert Linear layers E_m.
        router     (nn.Linear):    Gating network G(z) → (B, M) logits.
        classifier (nn.Linear):    Projection head C(h) → (B, num_classes).
    """

    def __init__(self, embed_dim: int, num_experts: int, num_classes: int):
        """
        Args:
            embed_dim:   Dimension of the input feature vector z from the featurizer.
            num_experts: Number of expert modules M.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.num_experts = num_experts

        # E_m: each expert is a single linear projection (simple yet interpretable)
        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)
        ])
        # G(z): routing network — produces un-normalised gating logits
        self.router = nn.Linear(embed_dim, num_experts)
        # C(h): final classification projection
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: Feature tensor from the featurizer, shape (B, embed_dim).

        Returns:
            logits: Class predictions (B, num_classes).
            pi:     Routing weights   (B, num_experts), sums to 1 over experts.
            h:      Blended features  (B, embed_dim).
        """
        # (B, M, D) — stack all M expert outputs along dim 1
        h_stack = torch.stack([expert(z) for expert in self.experts], dim=1)
        # (B, M) — normalised routing weights
        pi = F.softmax(self.router(z), dim=-1)
        # (B, D) — convex combination of expert outputs weighted by π
        h = (pi.unsqueeze(-1) * h_stack).sum(dim=1)
        # (B, C) — final classification
        logits = self.classifier(h)
        return logits, pi, h
