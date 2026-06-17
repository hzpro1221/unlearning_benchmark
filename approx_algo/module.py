import os
import time
import copy
import torch
import torch.nn.functional as F
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent
import inspect

from metric.fa import forget_acc

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
        # ablation config
        selection_option="diff",                
        update_scope="selected_experts_and_head", 
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
        
        self.selection_option = selection_option
        self.update_scope = update_scope

    def _loss_sparse(self, pi):
        entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
        return entropy.mean()

    def _loss_balance(self, pi, module_name, use_ema, ema_states, ema_alpha):
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

    # helper function for unlearning phase.
    # get forget and retain mass.
    def _get_routing_mass(self, loader):
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

    def _apply_update_scope(self, selected_experts_per_layer, moe_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        if self.update_scope == "full_model":
            for param in self.model.parameters():
                param.requires_grad = True
            return

        if self.update_scope == "all_experts":
            for m in moe_layers:
                for expert in m.experts:
                    for param in expert.parameters():
                        param.requires_grad = True
            return

        for l_idx, m in enumerate(moe_layers):
            selected_experts = selected_experts_per_layer[l_idx]
            for expert_idx, expert in enumerate(m.experts):
                if expert_idx in selected_experts:
                    for param in expert.parameters():
                        param.requires_grad = True

        if self.update_scope == "selected_experts_only":
            pass # already handled.

        elif self.update_scope == "selected_experts_and_head":
            for param in self.model.classifier_head.parameters():
                param.requires_grad = True

        elif self.update_scope == "selected_experts_and_router":
            for m in moe_layers:
                for param in m.router.parameters():
                    param.requires_grad = True

        elif self.update_scope == "selected_experts_and_last_block":
            last_block = self.model.featurizer.model.blocks[-1]
            for param in last_block.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown update_scope: {self.update_scope}")

    def _get_gradient_influence(self, data_loader):
        self.model.train()
        self.optimizer.zero_grad()
        
        for batch in data_loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            
            logits, _ = self.model.forward_with_grad(images)
            loss = self._unlearn_loss_forget(logits, labels)
            
            loss.backward()
            
        moe_layers = [m for _, m in self.model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']
        grad_scores = []
        
        for m in moe_layers:
            layer_scores = []
            for expert in m.experts:
                grad_mag = 0.0
                for param in expert.parameters():
                    if param.grad is not None:
                        grad_mag += param.grad.abs().sum().item()
                layer_scores.append(grad_mag)
            
            grad_scores.append(torch.tensor(layer_scores, device=self.device))
            
        self.optimizer.zero_grad() 
        return grad_scores

    def _unlearn_loss_forget(self, logits_f, labels_f):
        return -self.criteria(logits_f, labels_f)

    def _unlearn_loss_retain(self, logits_r, labels_r):
        return self.criteria(logits_r, labels_r)

    def _unlearn_loss_distill(self, logits_r, images_r, origin_model):
        with torch.no_grad():
            orig_logits_r, _ = origin_model.forward_with_grad(images_r)
            
        log_preds = F.log_softmax(logits_r, dim=-1)
        target_preds = F.softmax(orig_logits_r, dim=-1)
        return F.kl_div(log_preds, target_preds, reduction='batchmean')

    def _unlearn_loss_separation(self, moe_layers, selected_experts_per_layer):
        loss_sep = torch.tensor(0.0, device=self.device)
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
                    
                    # old version -> vanishing.
                    # inner_product = torch.matmul(Hm_tilde.t(), Hn_tilde) / batch_tokens_len
                    
                    # new version.
                    inner_product = torch.matmul(Hm_tilde.t(), Hn_tilde)
                    loss_sep += torch.norm(inner_product, p='fro') ** 2
        return loss_sep

    def learn(self, ckpt_path, ema_alpha=0.9):
        self.model._set_grad_mode("learning")
        use_ema = self.train_loader.batch_size <= 8
        ema_states = {}
        total_train_time = 0.0
        total_train_steps = 0 

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
                
                if len(all_pi) > 0:
                    sp_loss = sum([self._loss_sparse(pi) for pi in all_pi]) / len(all_pi)
                    bal_loss = sum([self._loss_balance(pi, n, use_ema, ema_states, ema_alpha) for pi, n in zip(all_pi, moe_names)]) / len(all_pi)
                    div_loss = sum([self._loss_diversity(h) for h in all_h]) / len(all_h)
                else:
                    sp_loss = torch.tensor(0.0, device=self.device)
                    bal_loss = torch.tensor(0.0, device=self.device)
                    div_loss = torch.tensor(0.0, device=self.device)
                
                t_loss = ce_loss + (self.lambda_sparse * sp_loss) + (self.lambda_balance * bal_loss) + (self.lambda_div * div_loss)

                t_loss.backward()
                self.optimizer.step()
                
                running_total += t_loss.item()
                running_ce += ce_loss.item()
                running_sp += sp_loss.item()
                running_bal += bal_loss.item()
                running_div += div_loss.item()
                
                total_train_steps += 1 

            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time
            
            num_batches = len(self.train_loader)
            avg_loss = running_total / num_batches
            
            print(f"epoch [{epoch+1}/{self.num_epoch}] | "
                  f"total_loss: {avg_loss:.4f} (ce: {running_ce/num_batches:.4f}, sp: {running_sp/num_batches:.4f}, bal: {running_bal/num_batches:.4f}, div: {running_div/num_batches:.4f}) | "
                  f"steps in epoch: {num_batches} | total_steps: {total_train_steps} | time: {epoch_train_time:.2f}s")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "ce_loss": running_ce / num_batches,
                "train_steps_accum": total_train_steps
            })
            
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

        print(f"[*] Training finished. Total Steps: {total_train_steps} | Running final evaluation...")
        fa_score, ra_score, ta_score, mia_score = self.evaluate()
        print(f"[Final Metrics] ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | ta: {ta_score*100:.2f}% | mia: {mia_score:.44f}")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_train_time_sec": total_train_time,
            "total_train_steps": total_train_steps,  
            "peak_memory_gb": peak_memory_gb,
            "retain_accuracy": ra_score,
            "forget_accuracy": fa_score,
            "test_accuracy": ta_score,
            "mia_score": mia_score
        })
        
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_train_time

    # this function for filtering retain loader.
    def _create_filtered_retain_loader(self, retain_loader, selected_experts_per_layer, moe_layers):
        self.model.eval()
        keep_indices = []
        current_idx = 0
        
        print("[Filter] Scanning retain set for expert intersection...")
        with torch.no_grad():
            for batch in retain_loader:
                images = batch[0].to(self.device)
                B_r = images.size(0)
                
                self.model(images)
                
                keep_mask = torch.zeros(B_r, dtype=torch.bool, device=self.device)
                
                for l_idx, m in enumerate(moe_layers):
                    selected_experts = selected_experts_per_layer[l_idx]
                    if not selected_experts:
                        continue
                        
                    _, topk_indices = m.last_pi_all.topk(m.gate_k, dim=-1)
                    S = topk_indices.size(0) // B_r
                    topk_indices = topk_indices.reshape(B_r, S, -1)
                    
                    for exp_idx in selected_experts:
                        keep_mask |= (topk_indices == exp_idx).any(dim=-1).any(dim=-1)
                
                true_indices = keep_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                keep_indices.extend((true_indices + current_idx).tolist())
                
                current_idx += B_r
                
        if len(keep_indices) == 0:
            print("[Filter] Warning: No retain samples matched the selected experts!")
            return None
            
        print(f"[Filter] Retained {len(keep_indices)} / {current_idx} samples.")
        
        subset = torch.utils.data.Subset(retain_loader.dataset, keep_indices)
        filtered_loader = torch.utils.data.DataLoader(
            subset, 
            batch_size=retain_loader.batch_size, 
            shuffle=True, 
            num_workers=retain_loader.num_workers if hasattr(retain_loader, 'num_workers') else 0,
            pin_memory=retain_loader.pin_memory if hasattr(retain_loader, 'pin_memory') else False
        )
        return filtered_loader

    def unlearn(self, fa_threshold, ckpt_path):
        origin_model = copy.deepcopy(self.model)
        origin_model.eval()
        for param in origin_model.parameters():
            param.requires_grad = False

        total_unlearn_time = 0.0
        total_unlearn_steps = 0
        early_stop = False
        
        closest_fa_score = float('inf') 

        if (self.selection_option != "diff"):
            print(f"[*] Starting unlearning with selection: {self.selection_option}, update_scope: {self.update_scope}")
        else:
            print(f"[*] Starting unlearning with selection: {self.selection_option}, alpha: {self.alpha} ,update_scope: {self.update_scope}")

        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()

            # expert selection (different options).
            selected_experts_per_layer = []
            if self.selection_option in ["diff", "ratio"]:
                forget_mass = self._get_routing_mass(self.forget_loader)
                retain_mass = self._get_routing_mass(self.retain_loader)
            elif self.selection_option == "gradient":
                grad_scores = self._get_gradient_influence(self.forget_loader)
            elif self.selection_option == "random":
                pass 
            else:
                raise ValueError(f"Invalid selection option: {self.selection_option}")
            
            moe_layers = [m for _, m in self.model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']

            for l_idx, m in enumerate(moe_layers):
                if self.selection_option == "random":
                    generator = torch.Generator(device=self.device).manual_seed(epoch + (l_idx * 100)) 
                    selected_experts = torch.randperm(m.num_experts, generator=generator, device=self.device)[:self.k_u].tolist()
                else:
                    if self.selection_option == "diff":
                        rho_m = forget_mass[l_idx] - self.alpha * retain_mass[l_idx]
                    elif self.selection_option == "ratio":
                        epsilon = 1e-8 
                        rho_m = forget_mass[l_idx] / (retain_mass[l_idx] + epsilon)
                    elif self.selection_option == "gradient":
                        rho_m = grad_scores[l_idx]
                    
                    if self.update_scope in ["all_experts", "full_model"]:
                        selected_experts = list(range(m.num_experts))
                    else:
                        _, topk_indices = rho_m.topk(self.k_u, dim=-1)
                        selected_experts = topk_indices.tolist()

                selected_experts_per_layer.append(selected_experts)
                m.allowed_experts = selected_experts

            # apply update scope.
            self._apply_update_scope(selected_experts_per_layer, moe_layers)

            # re-init optimizer.
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            opt_class = type(self.optimizer)
            valid_kwargs = inspect.signature(opt_class.__init__).parameters.keys()
            filtered_defaults = {k: v for k, v in self.optimizer.defaults.items() if k in valid_kwargs}
            self.optimizer = opt_class(trainable_params, **filtered_defaults)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[*] Update Scope: {self.update_scope} | Trainable Params: {trainable_params_count:,} / {total_params:,} ({(trainable_params_count / total_params) * 100:.2f}%)")

            # filter retain set.
            epoch_retain_loader = self._create_filtered_retain_loader(self.retain_loader, selected_experts_per_layer, moe_layers)

            self.model.train()
            origin_model.eval()
            total_loss_accum = 0.0
            
            if epoch_retain_loader is not None:
                retain_iter = iter(epoch_retain_loader)
            else:
                retain_iter = None
            
            for step, forget_batch in enumerate(self.forget_loader):
                
                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                # caculate loss forget & loss separation.
                logits_f, _ = self.model.forward_with_grad(images_f)
                loss_forget = self._unlearn_loss_forget(logits_f, labels_f)
                loss_sep = self._unlearn_loss_separation(moe_layers, selected_experts_per_layer)

                # only distill if retain loss is not empty.
                if retain_iter is not None:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(epoch_retain_loader)
                        retain_batch = next(retain_iter)
                        
                    images_r = retain_batch[0].to(self.device)
                    labels_r = retain_batch[1].to(self.device)
                    
                    logits_r, _ = self.model.forward_with_grad(images_r)
                    loss_retain = self._unlearn_loss_retain(logits_r, labels_r)
                    loss_distill = self._unlearn_loss_distill(logits_r, images_r, origin_model)
                else:
                    loss_retain = torch.tensor(0.0, device=self.device)
                    loss_distill = torch.tensor(0.0, device=self.device)

                total_loss = loss_forget + (self.beta * loss_retain) + (self.gamma * loss_distill) + (self.eta * loss_sep)
                total_loss.backward()
                
                self.optimizer.step()
                
                total_loss_accum += total_loss.item()
                total_unlearn_steps += 1
                
            avg_loss = total_loss_accum / len(self.forget_loader)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time

            print(f"[*] End of epoch {epoch+1}. Running full evaluation...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            self.model.train()
            
            if fa_threshold < fa_score < closest_fa_score:
                closest_fa_score = fa_score
                print(f"[!] New closest FA found at epoch end: {closest_fa_score*100:.2f}%. Saving fallback checkpoint...")
                torch.save(self.model.state_dict(), f"{ckpt_path}_closest_fa.pt")

            if fa_score <= fa_threshold:
                print(f"[*] Target condition met (FA = {fa_score*100:.2f}% <= {fa_threshold*100:.2f}%).")
                early_stop = True

            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | Steps: {total_unlearn_steps}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)
            
            wandb.log({
                "epoch": epoch+1, 
                "unlearn_loss": avg_loss, 
                "fa": fa_score, 
                "ra": ra_score, 
                "ta": ta_score, 
                "mia": mia_score,
                "unlearn_steps_accum": total_unlearn_steps  
            })
            
            torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

            if early_stop:
                break

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        
        wandb.log({
            "total_unlearn_time_sec": total_unlearn_time,
            "total_unlearn_steps": total_unlearn_steps,  
            "peak_memory_gb": peak_memory_gb
        })
        
        print(f"[*] Unlearning finished. Total Steps: {total_unlearn_steps} | Total Time: {total_unlearn_time:.2f}s")
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

        return total_unlearn_time