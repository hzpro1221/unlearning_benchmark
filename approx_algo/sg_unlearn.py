import os
import time
import torch
import torch.nn as nn
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class SG_Unlearning(Gradient_Ascent):
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
        alpha=0.6,  # Scales the IFT gradient step
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
        self.alpha = alpha
        self.proxy_loss_M = nn.BCEWithLogitsLoss()

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.train()
        total_unlearn_time = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_retain_loss = 0.0
            total_adv_loss = 0.0
            
            retain_iter = iter(self.retain_loader)
            unseen_iter = iter(self.unseen_loader)
            
            for forget_batch in self.forget_loader:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                    
                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)

                self.optimizer.zero_grad()
                logits_r, _ = self.model.forward_with_grad(images_r)
                loss_r = self.criteria(logits_r, labels_r) 
                
                loss_r.backward()
                self.optimizer.step() 
                total_retain_loss += loss_r.item()

                images_f = forget_batch[0].to(self.device)
                
                try:
                    unseen_batch = next(unseen_iter)
                except StopIteration:
                    unseen_iter = iter(self.unseen_loader)
                    unseen_batch = next(unseen_iter)
                images_u = unseen_batch[0].to(self.device)

                logits_f, _ = self.model.forward_with_grad(images_f)
                logits_u, _ = self.model.forward_with_grad(images_u)
                
                X_train = torch.cat([logits_f, logits_u], dim=0)
                Y_train = torch.cat([
                    torch.ones(logits_f.size(0), 1), 
                    torch.zeros(logits_u.size(0), 1)
                ], dim=0).to(self.device)

                lam = 1e-3
                logit_dim = X_train.size(1)
                I = torch.eye(logit_dim, device=self.device)
                
                # (X^T X + \lambda I) \theta_a = X^T Y
                A = X_train.t() @ X_train + lam * I
                B = X_train.t() @ Y_train
                
                theta_a_opt = torch.linalg.solve(A, B)

                self.optimizer.zero_grad()
                
                preds_f = logits_f @ theta_a_opt
                preds_u = logits_u @ theta_a_opt
                
                adv_labels_f = torch.zeros_like(preds_f)
                adv_labels_u = torch.zeros_like(preds_u)
                
                loss_M = self.proxy_loss_M(preds_f, adv_labels_f) + self.proxy_loss_M(preds_u, adv_labels_u)
                
                scaled_loss_M = self.alpha * loss_M
                scaled_loss_M.backward()
                
                self.optimizer.step() 
                total_adv_loss += loss_M.item()
                
            avg_retain_loss = total_retain_loss / len(self.forget_loader)
            avg_adv_loss = total_adv_loss / len(self.forget_loader)
            
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s")
            print(f"--> Retain Loss: {avg_retain_loss:.4f} | Proxy M Loss: {avg_adv_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)
            
            wandb.log({
                "epoch": epoch+1, 
                "unlearn_retain_loss": avg_retain_loss,
                "unlearn_adv_loss": avg_adv_loss,
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
        wandb.log({"total_train_time_sec": total_unlearn_time, "peak_memory_gb": peak_memory_gb})

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time