import time
import torch
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class SalUn(Gradient_Ascent):
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
        
        alpha=0.6, # regularization parameter for retain loss
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
        self.masks = {}

    def _compute_saliency_map(self):
        # calculate gradients of standard cross-entropy loss on the forgetting dataset 
        # use the pre-unlearning weights for identifying salient matrix.
        self.model.eval()
        self.optimizer.zero_grad()
        
        for batch in self.forget_loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            
            logits, _ = self.model.forward_with_grad(images)
            loss = self.criteria(logits, labels)
            loss.backward()
            
        # collect all absolute gradients into a single 1d tensor to find the global median
        all_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().abs().view(-1))
                
        if not all_grads:
            return
            
        all_grads = torch.cat(all_grads)
        gamma = all_grads.median()
        
        # apply hard thresholding: 1 if gradient >= gamma (salient), 0 otherwise (intact)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.masks[name] = (param.grad.detach().abs() >= gamma).float()
                
        self.optimizer.zero_grad()

    def unlearn(self, fa_threshold, ckpt_path):
        # construct the weight saliency map
        self._compute_saliency_map()
        
        self.model.train()
        total_unlearn_time = 0.0
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_loss = 0.0
            
            # setup infinite iterator for the retain set
            retain_iter = iter(self.retain_loader)
            
            for forget_batch in self.forget_loader:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                    
                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)
                
                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- random labeling on forget set ---
                logits_f, _ = self.model.forward_with_grad(images_f)
                num_classes = logits_f.shape[1]
                
                shifts = torch.randint(
                    low=1, 
                    high=num_classes, 
                    size=labels_f.shape, 
                    dtype=labels_f.dtype, 
                    device=self.device
                )
                random_labels = (labels_f + shifts) % num_classes
                loss_forget = self.criteria(logits_f, random_labels)
                
                # --- cross-entropy on retain set ---
                logits_r, _ = self.model.forward_with_grad(images_r)
                loss_retain = self.criteria(logits_r, labels_r)
                
                # combine losses
                batch_loss = loss_forget + (self.alpha * loss_retain)
                batch_loss.backward()
                
                # --- apply weight saliency mask ---
                # freeze intact weights by zeroing their gradients before the optimizer step
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self.masks:
                        param.grad.data *= self.masks[name]
                        
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