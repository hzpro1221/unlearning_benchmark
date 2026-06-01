# implements boundary expanding via shadow class remapping.
import os
import time
import torch
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class Boundary_Expanding(Gradient_Ascent):
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
        device="cuda",
        **kwargs 
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

    def unlearn(self, fa_threshold, ckpt_path):
        total_unlearn_time = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_loss = 0.0
            
            for batch in self.forget_loader:
                images = batch[0].to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()
                
                logits, _ = self.model.forward_with_grad(images)
                
                # phase 1: create the shadow class
                shadow_logits = torch.zeros(logits.size(0), 1).to(self.device)
                aug_logits = torch.cat([logits, shadow_logits], dim=1)
                
                # target the newly appended shadow class (index = original num_classes)
                num_original_classes = logits.size(1)
                shadow_labels = torch.full((logits.size(0),), num_original_classes, dtype=torch.long).to(self.device)
                
                # phase 2: boundary expanding update
                # forces original class logits to disperse and minimize against the shadow class
                loss = self.criteria(aug_logits, shadow_labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # --- EVALUATION & LOGGING ---
            avg_loss = total_loss / len(self.forget_loader)
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
            "total_train_time_sec": total_unlearn_time,
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time