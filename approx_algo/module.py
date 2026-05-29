import os
import time
import copy
import torch
import torch.nn.functional as F
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class Module(Gradient_Ascent):
    def __init__(
        self,
        model,
        train_loader,
        test_loader, 
        unseen_loader,
        forget_loader,
        forget_test_loader,
        retain_loader,
        retain_test_loader,
        optimizer,
        criteria,
        num_epoch,
        # config for learn
        lambda_sparse=1.0,
        lambda_balance=1.0,
        lambda_div=1.0,
        # config for unlearn
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        eta=1.0,
        k_u=2,
        device="cuda"
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader, 
            unseen_loader=unseen_loader,
            forget_loader=forget_loader,
            forget_test_loader=forget_test_loader,
            retain_loader=retain_loader,
            retain_test_loader=retain_test_loader,
            optimizer=optimizer,
            criteria=criteria,
            num_epoch=num_epoch,
            device=device
        )
        
        # verify architecture compatibility
        actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        supported_models = ['ModuleArchitecture']
        if actual_model.__class__.__name__ not in supported_models:
            raise TypeError(f"Module does not support {self.model.__class__.__name__}. Supported: {supported_models}")
            
        self.lambda_sparse = lambda_sparse
        self.lambda_balance = lambda_balance
        self.lambda_div = lambda_div
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.k_u = k_u

    # ==========================================
    # helper methods for learn phase
    # ==========================================
    def _loss_sparse(self, pi):
        # penalize entropy to enforce routing sparsity
        entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
        return entropy.mean()

    def _loss_balance(self, pi, module_name, use_ema, ema_states, ema_alpha):
        # ensure even token distribution across experts
        M = pi.size(-1)
        mean_pi = pi.mean(dim=0) 
        
        if use_ema:
            if module_name not in ema_states:
                ema_states[module_name] = torch.ones_like(mean_pi) / M
            effective_pi = ema_alpha * ema_states[module_name] + (1 - ema_alpha) * mean_pi
            ema_states[module_name] = effective_pi.detach()
        else:
            effective_pi = mean_pi

        return ((effective_pi - 1.0 / M) ** 2).sum()

    def _loss_diversity(self, h_stack, eps=1e-6):
        # penalize representation similarity between experts
        B, M, r = h_stack.shape
        if M < 2: 
            return h_stack.new_zeros(())
        
        loss = h_stack.new_zeros(1).squeeze()
        H_tilde = []
        for m in range(M):
            H_m = h_stack[:, m, :] 
            norm_F = H_m.norm(p='fro').clamp(min=eps)
            H_tilde.append(H_m / norm_F)
            
        for m in range(M):
            for n in range(M):
                if m == n: 
                    continue
                C = (H_tilde[m].T @ H_tilde[n])
                loss += (C ** 2).sum()
        return loss

    # ==========================================
    # helper methods for unlearn phase
    # ==========================================
    def _get_routing_mass(self, loader):
        # accumulate routing decisions over a dataset to identify expert specialization
        self.model.eval()
        masses = None
        total_tokens = 0
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                self.model(images)
                
                batch_masses = []
                num_tokens = 0
                for _, m in self.model.named_modules():
                    if m.__class__.__name__ == 'DeepMoELayer':
                        batch_masses.append(m.last_pi_all.sum(dim=0))
                        num_tokens = m.last_pi_all.size(0)
                        
                if masses is None:
                    masses = batch_masses
                else:
                    masses = [m + b for m, b in zip(masses, batch_masses)]
                total_tokens += num_tokens
                
        return [m / max(total_tokens, 1) for m in masses]

    def _unlearn_loss_forget(self, logits_f, labels_f):
        # maximize error on forget set
        return -self.criteria(logits_f, labels_f)

    def _unlearn_loss_retain(self, logits_r, labels_r):
        # minimize error on retain set
        return self.criteria(logits_r, labels_r)

    def _unlearn_loss_distill(self, logits_r, images_r, origin_model):
        # maintain global representations via distillation
        with torch.no_grad():
            orig_logits_r, _ = origin_model.forward_with_grad(images_r)
            
        log_preds = F.log_softmax(logits_r, dim=-1)
        target_preds = F.softmax(orig_logits_r, dim=-1)
        return F.kl_div(log_preds, target_preds, reduction='batchmean')

    def _unlearn_loss_separation(self, moe_layers, selected_experts_per_layer):
        # separate unlearned representations from frozen ones
        loss_sep = 0.0
        for l_idx, m in enumerate(moe_layers):
            H = m.last_h 
            batch_tokens_len = H.size(0)
            
            selected_M_f = selected_experts_per_layer[l_idx]
            frozen_M_r = [i for i in range(m.num_experts) if i not in selected_M_f]
            
            for expert_m in selected_M_f:
                for n in frozen_M_r:
                    Hm = H[:, expert_m, :] 
                    Hn = H[:, n, :] 
                    
                    norm_m = torch.norm(Hm, p='fro') + 1e-8
                    norm_n = torch.norm(Hn, p='fro') + 1e-8
                    
                    Hm_tilde = Hm / norm_m
                    Hn_tilde = Hn / norm_n
                    
                    inner_product = torch.matmul(Hm_tilde.t(), Hn_tilde) / batch_tokens_len
                    loss_sep += torch.norm(inner_product, p='fro') ** 2
        return loss_sep

    # ==========================================
    # core execution methods
    # ==========================================
    def learn(self, ckpt_path, ema_alpha=0.9):
        self.model._set_grad_mode("learning")
        use_ema = self.train_loader.batch_size <= 8
        ema_states = {}
        total_train_time = 0.0

        for epoch in range(self.num_epoch):
            self.model.train()
            epoch_start_time = time.time() 
            
            running_total, running_ce, running_sp, running_bal, running_div = 0.0, 0.0, 0.0, 0.0, 0.0

            for batch in self.train_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                
                all_pi, all_h, moe_names = [], [], []
                for name, module in self.model.featurizer.model.named_modules():
                    if module.__class__.__name__ == 'DeepMoELayer':
                        all_pi.append(module.last_pi)
                        all_h.append(module.last_h)
                        moe_names.append(name) 

                ce_loss = self.criteria(logits, labels)
                sp_loss = sum([self._loss_sparse(pi) for pi in all_pi]) / len(all_pi)
                bal_loss = sum([self._loss_balance(pi, n, use_ema, ema_states, ema_alpha) for pi, n in zip(all_pi, moe_names)]) / len(all_pi)
                div_loss = sum([self._loss_diversity(h) for h in all_h]) / len(all_h)
                
                t_loss = ce_loss + (self.lambda_sparse * sp_loss) + (self.lambda_balance * bal_loss) + (self.lambda_div * div_loss)

                t_loss.backward()
                self.optimizer.step()
                
                running_total += t_loss.item()
                running_ce += ce_loss.item()
                running_sp += sp_loss.item()
                running_bal += bal_loss.item()
                running_div += div_loss.item()

            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time
            
            num_batches = len(self.train_loader)
            avg_loss = running_total / num_batches
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"epoch [{epoch+1}/{self.num_epoch}] | "
                  f"total_loss: {avg_loss:.4f} (ce: {running_ce/num_batches:.4f}, sp: {running_sp/num_batches:.4f}, bal: {running_bal/num_batches:.4f}, div: {running_div/num_batches:.4f}) | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "ce_loss": running_ce / num_batches,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })
            
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_train_time_sec": total_train_time,
            "peak_memory_gb": peak_memory_gb
        })
        
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

        return total_train_time

    def unlearn(self, fa_threshold, ckpt_path):
        origin_model = copy.deepcopy(self.model)
        origin_model.eval()
        for param in origin_model.parameters():
            param.requires_grad = False

        total_unlearn_time = 0.0

        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()

            selected_experts_per_layer = []
            forget_mass = self._get_routing_mass(self.forget_loader)
            retain_mass = self._get_routing_mass(self.retain_loader)
            
            moe_layers = [m for _, m in self.model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']
            self.model._set_grad_mode("unlearning")

            for l_idx, m in enumerate(moe_layers):
                rho_m = forget_mass[l_idx] - self.alpha * retain_mass[l_idx]
                _, topk_indices = rho_m.topk(self.k_u, dim=-1)
                selected_experts = topk_indices.tolist()
                selected_experts_per_layer.append(selected_experts)
                
                for expert_idx, expert in enumerate(m.experts):
                    is_selected = expert_idx in selected_experts
                    for param in expert.parameters():
                        param.requires_grad = is_selected

            self.model.train()
            origin_model.eval()
            retain_iter = iter(self.retain_loader)
            
            total_loss_accum = 0.0
            
            for forget_batch in self.forget_loader:
                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)
                
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                    
                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                logits_f, _ = self.model.forward_with_grad(images_f)
                logits_r, _ = self.model.forward_with_grad(images_r)
                
                loss_forget = self._unlearn_loss_forget(logits_f, labels_f)
                loss_retain = self._unlearn_loss_retain(logits_r, labels_r)
                loss_distill = self._unlearn_loss_distill(logits_r, images_r, origin_model)
                loss_sep = self._unlearn_loss_separation(moe_layers, selected_experts_per_layer)

                total_loss = loss_forget + (self.beta * loss_retain) + (self.gamma * loss_distill) + (self.eta * loss_sep)
                total_loss.backward()
                self.optimizer.step()
                
                total_loss_accum += total_loss.item()
                
            avg_loss = total_loss_accum / len(self.forget_loader)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time

            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)
            
            wandb.log({
                "epoch": epoch+1, 
                "unlearn_loss": avg_loss, 
                "ra": ra_score, 
                "fa": fa_score, 
                "ta": ta_score, 
                "mia": mia_score
            })
            
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_unlearn_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb
        })
        
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

        return total_unlearn_time