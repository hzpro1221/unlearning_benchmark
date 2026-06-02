import time
import copy
import torch
import torch.nn.functional as F
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class ASU(Gradient_Ascent):
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

        tau=5.0,              
        support_weight=1.0,   
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
        self.tau = tau
        self.support_weight = support_weight

    def unlearn(self, fa_threshold, ckpt_path):
        forget_teacher = copy.deepcopy(self.model)
        forget_teacher.eval()
        for param in forget_teacher.parameters():
            param.requires_grad = False
            
        self.model.train()
        total_unlearn_time = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_loss = 0.0
            
            retain_iter = iter(self.retain_loader)
            
            for forget_batch in self.forget_loader:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                    
                images_f = forget_batch[0].to(self.device)
                
                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- support loss: cross-entropy on retain set ---
                logits_r, _ = self.model.forward_with_grad(images_r)
                loss_retain = self.criteria(logits_r, labels_r)
                
                # --- primary loss: kl divergence on forget set ---
                with torch.no_grad():
                    teacher_logits_f, _ = forget_teacher.forward_with_grad(images_f, tau=self.tau)
                    
                student_logits_f, _ = self.model.forward_with_grad(images_f)
                
                student_log_probs = F.log_softmax(student_logits_f, dim=-1)
                teacher_probs = F.softmax(teacher_logits_f, dim=-1)
                
                loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                
                batch_loss = loss_kl + (self.support_weight * loss_retain)
                
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                
            avg_loss = total_loss / len(self.forget_loader)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | Total Loss: {avg_loss:.4f}")
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
            "total_train_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time