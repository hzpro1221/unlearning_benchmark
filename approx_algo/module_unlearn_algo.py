import copy
import torch
import torch.nn.functional as F


class ModularUnlearning:
    """
    Surgical three-phase unlearning for MoEDeiTArchitecture.

    Expert selection uses forget-specific responsibility to prefer experts that
    are strongly activated by the forget set but not the retain set:

        ρ_m = s_m(D_f) − α · s_m(D_r)

    where s_m(D) = (1/|D|) Σ_{x∈D} π_m(x) is the mean routing mass over D.

    Expert m is activated in every transformer block simultaneously.
    The router, backbone, non-selected experts, and classifier remain frozen.

    Lifecycle
    ---------
    Phase 1  begin_modular_unlearn(forget_loader, retain_loader)
    Phase 2  update_modular_unlearn(forget_mb, retain_mb)  [per step]
    Phase 3  end_modular_unlearn()

    Required model interface
    ------------------------
        model._forward(x)          → (logits, pi, h)   pi shape (B, M), detached
        model.forward_with_grad(x) → (logits, features)
        model.inference(x)         → (logits, features)
        model.moe_adapters         — list[MoEAdapter], one per transformer block
        model.num_experts          — int M
    """

    def __init__(
        self,
        model,
        lr: float = 1e-4,
        beta: float = 1.0,
        gamma: float = 1.0,
        eta: float = 0.1,
        alpha_resp: float = 1.0,
        top_k: int = None,
        tau: float = None,
        device: str = "cuda",
    ):
        """
        Args:
            model:       MoEDeiTArchitecture instance.
            lr:          Learning rate for the targeted Adam optimizer.
            beta:        Weight on L_retain.
            gamma:       Weight on L_distill (KL from frozen teacher on retain).
            eta:         Weight on L_sep (retained expert separation). 0 = disabled.
            alpha_resp:  α in ρ_m = s_m(D_f) − α · s_m(D_r).
                         0 = raw forget routing mass; 1 = forget-minus-retain.
            top_k:       Activate the top-k highest-ρ experts. Priority over tau.
            tau:         Activate experts with ρ_m > tau.
                         Falls back to top-1 if none qualify.
            device:      'cuda' or 'cpu'.

        If neither top_k nor tau is given, defaults to top max(1, M // 4).
        """
        if not hasattr(model, '_forward') or not hasattr(model, 'moe_adapters'):
            raise TypeError(
                "model must expose ._forward() and .moe_adapters "
                "(use MoEDeiTArchitecture, not DeiTArchitecture)"
            )

        self.model = model
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.alpha_resp = alpha_resp
        self.top_k = top_k
        self.tau = tau
        self.device = device

        self._teacher = None
        self._optimizer = None
        self._active_indices: list = []
        self._retain_indices: list = []

    # ------------------------------------------------------------------
    # Phase 1 — Setup
    # ------------------------------------------------------------------

    def begin_modular_unlearn(self, forget_loader, retain_loader) -> dict:
        """
        Computes forget-specific responsibility ρ_m for every expert,
        selects M_f, freezes all params except expert[m] for m ∈ M_f
        across every transformer block, snapshots a frozen teacher, and
        creates a targeted Adam optimizer.

        Args:
            forget_loader: DataLoader yielding (images, labels [, domain]).
            retain_loader: DataLoader yielding (images, labels [, domain]).

        Returns:
            dict:
                'expert_indices'   — sorted list of selected expert indices M_f.
                'scores_forget'    — s_m(D_f) per expert (list, index = expert).
                'scores_retain'    — s_m(D_r) per expert (list, index = expert).
                'responsibility'   — ρ_m per expert (list, index = expert).
        """
        M = self.model.num_experts

        # --- 1a. score experts on forget and retain sets ---
        scores_f = self._routing_mass(forget_loader)   # (M,) s_m(D_f)
        scores_r = self._routing_mass(retain_loader)   # (M,) s_m(D_r)

        # --- 1b. forget-specific responsibility: ρ_m = s_m(D_f) - α·s_m(D_r) ---
        rho = scores_f - self.alpha_resp * scores_r    # (M,)

        # --- 1c. select M_f ---
        self._active_indices = self._select_experts(rho)
        self._retain_indices = [m for m in range(M) if m not in set(self._active_indices)]

        # --- 1d. freeze all params; unfreeze expert[m] in every block for m ∈ M_f ---
        for param in self.model.parameters():
            param.requires_grad_(False)
        for adapter in self.model.moe_adapters:
            for m in self._active_indices:
                for param in adapter.experts[m].parameters():
                    param.requires_grad_(True)

        # --- 1e. frozen teacher snapshot ---
        self._teacher = copy.deepcopy(self.model)
        for param in self._teacher.parameters():
            param.requires_grad_(False)
        self._teacher.eval()

        # --- 1f. optimizer over active expert params only ---
        active_params = [
            p
            for adapter in self.model.moe_adapters
            for m in self._active_indices
            for p in adapter.experts[m].parameters()
        ]
        self._optimizer = torch.optim.Adam(active_params, lr=self.lr)

        return {
            "expert_indices":  self._active_indices,
            "scores_forget":   scores_f.tolist(),
            "scores_retain":   scores_r.tolist(),
            "responsibility":  rho.tolist(),
        }

    # ------------------------------------------------------------------
    # Phase 2 — Per-step update
    # ------------------------------------------------------------------

    def update_modular_unlearn(self, minibatch_forget, minibatch_retain) -> dict:
        """
        One gradient step of the modular unlearning objective:

            L_forget  = -CE(model(x_f), y_f)            [gradient ascent]
            L_retain  =  CE(model(x_r), y_r)             [preserve accuracy]
            L_distill =  KL(p_old(y|x_r) ‖ p_new(y|x_r)) [output stability]
            L_sep     =  Σ_{m∈M_f} Σ_{n∈M_r}
                             ‖(1/B) H̃_m(x_r)^T H̃_n(x_r)‖_F²  [expert separation]

            L_unlearn = L_forget + β·L_retain + γ·L_distill + η·L_sep

        L_sep is computed from adapter.last_expert_outputs which is populated
        by the retain forward pass. Gradients flow through H_m (active experts)
        but not through H_n (frozen experts — their params have requires_grad=False).

        Args:
            minibatch_forget: tuple (images, labels [, domain]) from forget_loader.
            minibatch_retain: tuple (images, labels [, domain]) from retain_loader.

        Returns:
            dict of scalar loss components: 'total', 'l_forget', 'l_retain',
            'l_distill', 'l_sep'.
        """
        x_f = minibatch_forget[0].to(self.device)
        y_f = minibatch_forget[1].to(self.device)
        x_r = minibatch_retain[0].to(self.device)
        y_r = minibatch_retain[1].to(self.device)

        self._optimizer.zero_grad()

        # L_forget — gradient ascent on forget set
        logits_f, _ = self.model.forward_with_grad(x_f)
        l_forget = -F.cross_entropy(logits_f, y_f)

        # L_retain — standard CE on retain set
        # after this call, adapter.last_expert_outputs holds retain batch outputs
        logits_r, _ = self.model.forward_with_grad(x_r)
        l_retain = F.cross_entropy(logits_r, y_r)

        # L_distill — KL(p_old ‖ p_new) on retain set
        teacher_logits_r, _ = self._teacher.inference(x_r)
        l_distill = F.kl_div(
            F.log_softmax(logits_r, dim=-1),
            F.softmax(teacher_logits_r.detach(), dim=-1),
            reduction="batchmean",
        )

        # L_sep — retained expert separation (computed from retain batch outputs)
        l_sep = torch.tensor(0.0, device=self.device)
        if self.eta > 0.0 and self._retain_indices:
            l_sep = self._sep_loss()

        total = (
            l_forget
            + self.beta    * l_retain
            + self.gamma   * l_distill
            + self.eta     * l_sep
        )

        total.backward()
        self._optimizer.step()

        return {
            "total":     total.item(),
            "l_forget":  l_forget.item(),
            "l_retain":  l_retain.item(),
            "l_distill": l_distill.item(),
            "l_sep":     l_sep.item(),
        }

    # ------------------------------------------------------------------
    # Phase 3 — Teardown
    # ------------------------------------------------------------------

    def end_modular_unlearn(self):
        """Releases teacher and optimizer; restores requires_grad globally."""
        del self._teacher
        self._teacher = None
        self._optimizer = None
        self._active_indices = []
        self._retain_indices = []
        for param in self.model.parameters():
            param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _routing_mass(self, loader) -> torch.Tensor:
        """
        Computes mean routing mass per expert over the given loader:
            s_m(D) = (1/|D|) Σ_{x∈D} π_m(x)

        π_m(x) is the per-expert routing probability averaged over all
        transformer blocks and token positions (returned by model._forward).

        Returns:
            scores: (M,) tensor, detached.
        """
        M = self.model.num_experts
        scores = torch.zeros(M, device=self.device)
        n_samples = 0

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                _logits, pi, _h = self.model._forward(x)   # pi: (B, M), detached
                scores += pi.sum(dim=0)
                n_samples += pi.shape[0]

        return scores / max(n_samples, 1)

    def _select_experts(self, rho: torch.Tensor) -> list:
        """
        Returns sorted expert indices M_f from responsibility scores ρ.

        Priority: top_k → tau → default top-max(1, M//4).
        """
        M = self.model.num_experts

        if self.top_k is not None:
            indices = rho.topk(min(self.top_k, M)).indices.tolist()

        elif self.tau is not None:
            indices = (rho > self.tau).nonzero(as_tuple=True)[0].tolist()
            if not indices:
                indices = [int(rho.argmax().item())]

        else:
            k = max(1, M // 4)
            indices = rho.topk(k).indices.tolist()

        return sorted(indices)

    def _sep_loss(self, eps: float = 1e-8) -> torch.Tensor:
        """
        Retained expert separation loss.

        For each transformer block and each active/frozen expert pair (m, n)
        with m ∈ M_f, n ∈ M_r:

            L_sep += ‖(1/B) H̃_m(x_r)^T H̃_n(x_r)‖_F²

        H_m = adapter.last_expert_outputs[:, m, :]  ∈ R^{B×D}
        H̃_m = H_m / (‖H_m‖_F + ε)

        Gradient flows through H_m (active expert, requires_grad=True).
        H_n has no grad (frozen expert params → no leaf tensor requires grad).
        """
        total = torch.tensor(0.0, device=self.device)
        M_f = self._active_indices
        M_r = self._retain_indices

        for adapter in self.model.moe_adapters:
            if adapter.last_expert_outputs is None:
                continue
            H = adapter.last_expert_outputs  # (B, M, D), populated by retain forward
            B = H.shape[0]

            for m in M_f:
                H_m = H[:, m, :]                                    # (B, D), with grad
                H_m_n = H_m / (H_m.norm() + eps)
                for n in M_r:
                    H_n = H[:, n, :]                                # (B, D), no grad
                    H_n_n = H_n / (H_n.norm() + eps)
                    cross = (H_m_n.T @ H_n_n) / B                  # (D, D)
                    total = total + cross.pow(2).sum()

        return total
