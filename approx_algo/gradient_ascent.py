import os
import torch
import wandb

from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia

class Gradient_Ascent:
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
        device="cuda"
    ):
        self.model = model
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.unseen_loader = unseen_loader

        self.forget_loader = forget_loader
        self.forget_test_loader = forget_test_loader

        self.retain_loader = retain_loader
        self.retain_test_loader = retain_test_loader

        self.optimizer = optimizer
        self.criteria = criteria

        self.num_epoch = num_epoch
        self.device = device
        
    def learn(self, ckpt_path):
        self.model.train()
        
        for epoch in range(self.num_epoch):
            total_loss = 0.0
            
            for batch in self.train_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                loss = self.criteria(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(f"epoch [{epoch+1}/{self.num_epoch}] | loss: {avg_loss:.4f} | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score
            })

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.train()
        
        for epoch in range(self.num_epoch):
            total_loss = 0.0
            
            for batch in self.forget_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                loss = self.criteria(logits, labels)
                
                # reverses the loss sign to perform gradient ascent
                # the optimizer will try to minimize (-loss), which effectively maximizes the actual (loss)
                ascent_loss = -loss
                ascent_loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.forget_loader)
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Loss: {avg_loss:.4f}")
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
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

    def evaluate(self):
        fa_score = forget_acc(self.model, self.forget_test_loader, self.device)
        ra_score = retain_acc(self.model, self.retain_test_loader, self.device)
        ta_score = test_acc(self.model, self.test_loader, self.device)
        mia_score = mia(self.model, self.unseen_loader, self.forget_test_loader, self.device)
        return fa_score, ra_score, ta_score, mia_score