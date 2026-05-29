import os
import time
import torch
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class L1_Sparse(Gradient_Ascent):
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
        # inherit base configurations from gradient ascent
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
        # execute unlearning via gradient ascent and l1 regularization
        self.model.train()
        total_unlearn_time = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_loss = 0.0
            
            for batch in self.forget_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                ce_loss = self.criteria(logits, labels)
                
                # compute l1 norm for non-normalization weights
                l1_penalty = 0.0
                num_params = 0 
                
                for name, param in self.model.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'norm' not in name:
                        l1_penalty += torch.sum(torch.abs(param))
                        num_params += param.numel() 
                
                if num_params > 0:
                    l1_penalty = l1_penalty / num_params
                
                # ascent on cross-entropy to forget, descent on l1 penalty for sparsity
                batch_loss = -(ce_loss) + (self.alpha * l1_penalty)
                
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                
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
            
            # halt unlearning if forget accuracy target is reached
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