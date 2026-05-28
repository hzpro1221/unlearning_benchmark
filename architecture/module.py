import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """
    A single expert — 2-layer MLP matching the DeiT block FFN structure.

        z → fc1 → GELU → drop → fc2 → drop → h_m
    """

    def __init__(self, embed_dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEAdapter(nn.Module):
    """
    Replaces the FFN (mlp) inside a DeiT transformer block.

    Mathematical flow per token position:
        π(z) = softmax(G(z))       routing weights   (B, T, M)
        h_m  = E_m(z)              expert output     (B, T, D)
        h    = Σ_m π_m · h_m      blended output    (B, T, D)

    After every forward pass two tensors are stored as attributes:

        last_routing_weights  (B, T, M)  — kept WITH gradient for L_sp.
                                           Callers that only need values
                                           should use .detach() themselves.

        last_expert_outputs   (B, M, D)  — token-averaged expert outputs,
                                           kept WITH gradient for L_div
                                           (training) and L_sep (unlearning).

    Attributes:
        experts  (nn.ModuleList): M ExpertFFN modules.
        router   (nn.Linear):    Gating network G — Linear(D, M).
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_experts: int, drop: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertFFN(embed_dim, hidden_dim, drop) for _ in range(num_experts)
        ])
        self.router = nn.Linear(embed_dim, num_experts)
        self.last_routing_weights: torch.Tensor = None   # (B, T, M), with grad
        self.last_expert_outputs: torch.Tensor = None    # (B, M, D), with grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        pi = F.softmax(self.router(x), dim=-1)               # (B, T, M)
        self.last_routing_weights = pi                        # keep grad for L_sp
        h_stack = torch.stack([e(x) for e in self.experts], dim=2)  # (B, T, M, D)
        self.last_expert_outputs = h_stack.mean(dim=1)        # (B, M, D), keep grad
        return (pi.unsqueeze(-1) * h_stack).sum(dim=2)        # (B, T, D)
