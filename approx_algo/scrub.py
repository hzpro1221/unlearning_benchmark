import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class SCRUB(Gradient_Ascent):
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
        alpha=0.1,
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

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.train()
        
        # create a frozen teacher model to retain original knowledge reference
        teacher_model = copy.deepcopy(self.model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        # kl divergence measures how student predictions deviate from the teacher
        kl_criterion = nn.KLDivLoss(reduction="batchmean")

        total_unlearn_time = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_forget_loss = 0.0
            total_retain_loss = 0.0
            
            # parallel iterator for the retain set
            retain_iter = iter(self.retain_loader)
            
            for forget_batch in self.forget_loader:
                # -- MAX-STEP: maximize divergence on forget set --
                images_f = forget_batch[0].to(self.device)
                
                self.optimizer.zero_grad()
                
                student_logits_f, _ = self.model.forward_with_grad(images_f)
                
                with torch.no_grad():
                    teacher_logits_f, _ = teacher_model.forward_with_grad(images_f)
                
                # kldivloss expects log-probs for input (student) and probs for target (teacher)
                log_prob_s_f = F.log_softmax(student_logits_f, dim=1)
                prob_t_f = F.softmax(teacher_logits_f, dim=1)
                
                # negative kldiv forces the student to forget by diverging from teacher
                loss_max = -kl_criterion(log_prob_s_f, prob_t_f)
                
                loss_max.backward()
                self.optimizer.step()
                total_forget_loss += loss_max.item()
                
                # -- MIN-STEP: preserve fidelity on retain set --
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    # loop retain_loader if it exhausts before forget_loader
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                    
                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                student_logits_r, _ = self.model.forward_with_grad(images_r)
                
                with torch.no_grad():
                    teacher_logits_r, _ = teacher_model.forward_with_grad(images_r)
                
                # standard task loss to maintain accuracy on retained data
                task_loss = self.criteria(student_logits_r, labels_r)
                
                # positive kldiv forces student to mimic teacher on retained data
                log_prob_s_r = F.log_softmax(student_logits_r, dim=1)
                prob_t_r = F.softmax(teacher_logits_r, dim=1)
                kl_loss_min = kl_criterion(log_prob_s_r, prob_t_r)
                
                loss_min = task_loss + kl_loss_min
                
                loss_min.backward()
                self.optimizer.step()
                total_retain_loss += loss_min.item()
                
            # -- EVALUATION & LOGGING --
            avg_forget_loss = total_forget_loss / len(self.forget_loader)
            avg_retain_loss = total_retain_loss / len(self.forget_loader)
            
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s")
            print(f"--> Forget Loss: {avg_forget_loss:.4f} | Retain Loss: {avg_retain_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)
            
            wandb.log({
                "epoch": epoch+1, 
                "unlearn_forget_loss": avg_forget_loss,
                "unlearn_retain_loss": avg_retain_loss,
                "ra": ra_score, 
                "fa": fa_score, 
                "ta": ta_score, 
                "mia": mia_score
            })
            
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")
            
            # halt training if target forget accuracy is met
            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_train_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time