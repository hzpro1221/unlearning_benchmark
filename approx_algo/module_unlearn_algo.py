import copy
import torch
import torch.nn.functional as F


class ModularUnlearning:
    """
    Surgical three-phase unlearning for MoEDeiTArchitecture.

    Exploits the model's explicit routing weights to identify and isolate the
    subset of experts M_f ⊆ {0…M-1} most activated by the forget set.
    Only those experts receive gradient updates; the backbone, router, classifier,
    and remaining experts are fully frozen throughout the unlearning process.

    Lifecycle
    ---------
    Phase 1  begin_modular_unlearn(forget_loader)
             → score experts via routing mass, select M_f, freeze rest,
               snapshot teacher, create targeted optimizer.

    Phase 2  update_modular_unlearn(forget_mb, retain_mb)   [called per step]
             → one gradient step of:
               L = L_forget + β·L_retain + γ·L_distill + λ_div·L_div

    Phase 3  end_modular_unlearn()
             → release teacher and optimizer, restore requires_grad globally.

    Model interface requirements
    ----------------------------
        model._forward(x)          → (logits, pi, h)   pi shape (B, M)
        model.forward_with_grad(x) → (logits, features)
        model.inference(x)         → (logits, features)
        model.moe_head.experts     — nn.ModuleList of M expert modules
        model.num_experts          — int M
    """

    def __init__(
        self,
        model,
        lr: float = 1e-4,
        beta: float = 1.0,
        gamma: float = 1.0,
        lambda_div: float = 0.0,
        top_k: int = None,
        tau: float = None,
        device: str = "cuda",
    ):
        """
        Args:
            model:      MoEDeiTArchitecture instance.
            lr:         Learning rate for the targeted optimizer.
            beta:       Weight on L_retain.
            gamma:      Weight on L_distill.
            lambda_div: Weight on L_div (0 = disabled).
            top_k:      Activate the top-k highest-scored experts.
                        Mutually exclusive with tau; top_k takes priority.
            tau:        Activate every expert whose routing-mass score > tau.
                        Falls back to top-1 if no expert exceeds the threshold.
            device:     'cuda' or 'cpu'.

        Selection priority:
            top_k given → select top-k
            tau given   → select score > tau  (fallback to top-1)
            neither     → default to top max(1, M // 4)
        """
        if not hasattr(model, '_forward') or not hasattr(model, 'moe_head'):
            raise TypeError(
                "model must expose ._forward() and .moe_head "
                "(use MoEDeiTArchitecture, not standard DeiTArchitecture)"
            )

        self.model = model
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.lambda_div = lambda_div
        self.top_k = top_k
        self.tau = tau
        self.device = device

        # populated in begin_modular_unlearn, cleared in end_modular_unlearn
        self._teacher = None
        self._optimizer = None
        self._active_indices: list = []

    # ------------------------------------------------------------------
    # Phase 1 — Setup
    # ------------------------------------------------------------------

    def begin_modular_unlearn(self, forget_loader) -> dict:
        """
        Scores all M experts on the forget set, selects M_f, freezes the
        non-selected components, builds a frozen teacher, and creates a
        targeted Adam optimizer over the active expert parameters only.

        Args:
            forget_loader: DataLoader yielding (images, labels [, domain]) tuples.

        Returns:
            dict:
                'expert_indices' — sorted list of selected expert indices in M_f.
                'scores'         — list of per-expert routing-mass scores (index = expert).
        """
        # --- 1a. score experts on forget set via routing weights ---
        scores = self._score_experts(forget_loader)

        # --- 1b. select M_f ---
        self._active_indices = self._select_experts(scores)

        # --- 1c. freeze everything; unfreeze M_f experts only ---
        for param in self.model.parameters():
            param.requires_grad_(False)
        for idx in self._active_indices:
            for param in self.model.moe_head.experts[idx].parameters():
                param.requires_grad_(True)

        # --- 1d. frozen teacher snapshot for distillation ---
        self._teacher = copy.deepcopy(self.model)
        for param in self._teacher.parameters():
            param.requires_grad_(False)
        self._teacher.eval()

        # --- 1e. optimizer over active expert params only ---
        active_params = [
            p
            for idx in self._active_indices
            for p in self.model.moe_head.experts[idx].parameters()
        ]
        self._optimizer = torch.optim.Adam(active_params, lr=self.lr)

        return {
            "expert_indices": self._active_indices,
            "scores": scores.tolist(),
        }

    # ------------------------------------------------------------------
    # Phase 2 — Per-step update
    # ------------------------------------------------------------------

    def update_modular_unlearn(self, minibatch_forget, minibatch_retain) -> dict:
        """
        One gradient step of the modular unlearning objective:

            L_forget  = -CE(model(x_f), y_f)          [gradient ascent on forget]
            L_retain  =  CE(model(x_r), y_r)           [preserve retain performance]
            L_distill =  KL(teacher(x_r) ‖ model(x_r)) [anchor to original knowledge]
            L_div     =  Frobenius decorrelation of M_f weight matrices  [optional]

            L_total = L_forget + β·L_retain + γ·L_distill + λ_div·L_div

        Args:
            minibatch_forget: tuple (images, labels [, domain]) from forget_loader.
            minibatch_retain: tuple (images, labels [, domain]) from retain_loader.

        Returns:
            dict of scalar loss components (Python floats):
                'total', 'l_forget', 'l_retain', 'l_distill', 'l_div'
        """
        x_f = minibatch_forget[0].to(self.device)
        y_f = minibatch_forget[1].to(self.device)
        x_r = minibatch_retain[0].to(self.device)
        y_r = minibatch_retain[1].to(self.device)

        self._optimizer.zero_grad()

        # L_forget — negative CE on forget set (gradient ascent)
        logits_f, _ = self.model.forward_with_grad(x_f)
        l_forget = -F.cross_entropy(logits_f, y_f)

        # L_retain — standard CE on retain set
        logits_r, _ = self.model.forward_with_grad(x_r)
        l_retain = F.cross_entropy(logits_r, y_r)

        # L_distill — KL(teacher_probs ‖ student_probs) on retain set
        teacher_logits_r, _ = self._teacher.inference(x_r)
        l_distill = F.kl_div(
            F.log_softmax(logits_r, dim=-1),
            F.softmax(teacher_logits_r.detach(), dim=-1),
            reduction="batchmean",
        )

        # L_div — optional Frobenius decorrelation across active expert weights
        l_div = torch.tensor(0.0, device=self.device)
        if self.lambda_div > 0.0:
            l_div = self._diversity_loss()

        total = (
            l_forget
            + self.beta * l_retain
            + self.gamma * l_distill
            + self.lambda_div * l_div
        )

        total.backward()
        self._optimizer.step()

        return {
            "total":     total.item(),
            "l_forget":  l_forget.item(),
            "l_retain":  l_retain.item(),
            "l_distill": l_distill.item(),
            "l_div":     l_div.item(),
        }

    # ------------------------------------------------------------------
    # Phase 3 — Teardown
    # ------------------------------------------------------------------

    def end_modular_unlearn(self):
        """
        Releases the teacher copy and optimizer, then restores requires_grad=True
        globally across all model parameters.
        """
        del self._teacher
        self._teacher = None
        self._optimizer = None
        self._active_indices = []

        for param in self.model.parameters():
            param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_experts(self, forget_loader) -> torch.Tensor:
        """
        Accumulates routing mass per expert over the forget set:
            scores[m] += Σ_{x in batch} π_m(x)

        Uses eval mode and no_grad since we are only accumulating statistics,
        not computing gradients for a training step.

        Returns:
            scores: (M,) tensor of accumulated routing mass per expert.
        """
        M = self.model.num_experts
        scores = torch.zeros(M, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in forget_loader:
                x = batch[0].to(self.device)
                _logits, pi, _h = self.model._forward(x)
                # pi: (B, M) — sum over batch dimension
                scores += pi.sum(dim=0)

        return scores

    def _select_experts(self, scores: torch.Tensor) -> list:
        """
        Returns a sorted list of expert indices to activate (M_f).

        Selection priority:
            top_k set  → select indices of top-k scores
            tau set    → select indices where score > tau; fallback to argmax
            neither    → default to top max(1, M // 4) experts
        """
        M = self.model.num_experts

        if self.top_k is not None:
            k = min(self.top_k, M)
            indices = scores.topk(k).indices.tolist()

        elif self.tau is not None:
            indices = (scores > self.tau).nonzero(as_tuple=True)[0].tolist()
            if not indices:
                indices = [int(scores.argmax().item())]

        else:
            # default: top M/4 experts
            k = max(1, M // 4)
            indices = scores.topk(k).indices.tolist()

        return sorted(indices)

    def _diversity_loss(self) -> torch.Tensor:
        """
        Frobenius decorrelation penalty on the weight matrices of M_f experts.
        Encourages selected experts to maintain diverse representations.

            L_div = mean |corr(w_i, w_j)| for all i ≠ j in M_f

        where corr(w_i, w_j) = (w_i/‖w_i‖) · (w_j/‖w_j‖).
        """
        weight_vecs = []
        for idx in self._active_indices:
            for name, param in self.model.moe_head.experts[idx].named_parameters():
                if 'weight' in name and param.requires_grad:
                    v = param.view(-1).float()
                    weight_vecs.append(v / v.norm(2).clamp(min=1e-8))

        if len(weight_vecs) < 2:
            return torch.tensor(0.0, device=self.device)

        stack = torch.stack(weight_vecs)          # (G, D_flat)
        gram = stack @ stack.T                    # (G, G) pairwise cosine similarities
        mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=self.device)
        return gram[mask].abs().mean()
