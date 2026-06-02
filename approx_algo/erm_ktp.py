# implements the erm pre-training and ktp unlearning algorithm (ckt, fkt, gkt).
import time
import copy
import torch
import torch.nn.functional as F
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent

class ERM_KTP(Gradient_Ascent):
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
        forget_classes=[0], 
        lambda_1=1.0, 
        lambda_2=1.0, 
        lambda_3=0.1, 
        warmup_epochs=0,
        mask_period=3,
        mask_epoch_min=2,
        device="cuda",
        **kwargs
    ):
        actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        if actual_model.__class__.__name__ != 'ERM_KTP_Resnet':
            raise TypeError(f"ERM_KTP exclusively supports ERM_KTP_Resnet. Received: {actual_model.__class__.__name__}")

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
        self.forget_classes = forget_classes
        
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.warmup_epochs = warmup_epochs
        self.mask_period = mask_period
        self.mask_epoch_min = mask_epoch_min

    def learn(self, ckpt_path):
        self.model.train()
        total_train_time = 0.0
        
        for epoch in range(self.num_epoch):
            total_loss = 0.0
            
            total_ce = 0.0
            total_l1 = 0.0
            total_ip = 0.0
            
            epoch_start_time = time.time()
            
            is_erm_epoch = (epoch >= self.warmup_epochs) and (epoch % self.mask_period >= self.mask_epoch_min)
            path_name = "ERM Path" if is_erm_epoch else "STD Path"
            
            for batch in self.train_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                if is_erm_epoch:
                    # erm path: ce loss + l1 sparsity + l2 orthogonality
                    logits, _, l1_reg, inner_product = self.model.forward_with_grad(images, labels=labels)
                    ce_loss = self.criteria(logits, labels)
                    
                    total_ce += ce_loss.item()
                    total_l1 += l1_reg.item()
                    total_ip += inner_product.item()
                    
                    loss = (self.lambda_1 * ce_loss) + (self.lambda_2 * l1_reg) + (self.lambda_3 * inner_product)
                    loss.backward()
                    self.optimizer.step()
                    
                    # enforces max value of each mask row to be 1
                    self.model.clip_lmask() 
                else:
                    # standard path: normal forward without masking layer
                    logits, _ = self.model.forward_with_grad(images)
                    loss = self.criteria(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                
            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time
                
            avg_loss = total_loss / len(self.train_loader)
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(f"epoch [{epoch+1}/{self.num_epoch}] ({path_name}) | loss: {avg_loss:.4f} | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s")
                  
            if is_erm_epoch:
                num_batches = len(self.train_loader)
                print(f"   -> [DEBUG ERM LOSSES] CE: {total_ce/num_batches:.4f} | L1 Sparsity: {total_l1/num_batches:.4f} | L2 Orthogonal: {total_ip/num_batches:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "retain_accuracy": ra_score,
                "forget_accuracy": fa_score,
                "test_accuracy": ta_score,
                "mia_score": mia_score,
                "is_erm_path": int(is_erm_epoch)
            })

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "total_train_time_sec": total_train_time,
            "peak_memory_gb": peak_memory_gb
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_train_time

    def execute_FKT_and_GKT(self, model_T, model_S):
        with torch.no_grad():
            bar_Gc = model_T.featurizer.lmask.get_bar_Gc(self.forget_classes)
            
            # --- FKT ---
            A_beta = bar_Gc.expand(model_S.num_classes, -1).to(self.device)
            
            A_gamma = torch.ones(model_S.num_classes).to(self.device)
            for c in self.forget_classes:
                A_gamma[c] = 0.0
            
            model_S.classifier_head.weight.data = model_T.classifier_head.weight.data * A_beta
            if model_S.classifier_head.bias is not None:
                model_S.classifier_head.bias.data = model_T.classifier_head.bias.data * A_gamma
                
            for param in model_S.classifier_head.parameters():
                param.requires_grad = False

            # --- GKT ---
            A_G = torch.ones_like(model_T.featurizer.lmask.mask).to(self.device)
            for c in self.forget_classes:
                A_G[:, c] = 0.0
                
            model_S.featurizer.lmask.mask.data = model_T.featurizer.lmask.mask.data * A_G
            model_S.featurizer.lmask.mask.requires_grad = False 

    def unlearn(self, fa_threshold, ckpt_path):
        total_unlearn_time = 0.0
        
        # setup teacher model (frozen)
        model_T = copy.deepcopy(self.model)
        model_T.eval()
        for param in model_T.parameters():
            param.requires_grad = False
            
        # setup student model (self.model)
        model_S = self.model
        
        # apply fkt and gkt before feature training
        self.execute_FKT_and_GKT(model_T, model_S)
        
        model_S.train()
        
        # only update the convolutional feature extractor
        optimizer_CKT = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_S.parameters()), 
            lr=self.optimizer.param_groups[0]['lr']
        )
        
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            total_loss = 0.0
            
            # --- CKT ---
            for batch in self.retain_loader:
                images = batch[0].to(self.device)
                optimizer_CKT.zero_grad()
                
                # teacher forward: f(x; theta_T) 
                with torch.no_grad():
                    f_T_conv = model_T.featurizer.feature_extractor(images)
                    f_T_pooled = torch.flatten(model_T.featurizer.avgpool(f_T_conv), 1)
                
                # student forward: g(x; theta_S) * -Gc
                g_S_conv = model_S.featurizer.feature_extractor(images)
                
                g_S_masked = model_S.featurizer.lmask.channel_express(g_S_conv, self.forget_classes)
                
                g_S_pooled = torch.flatten(model_S.featurizer.avgpool(g_S_masked), 1)
                
                # mse loss: MSE(f(x), g(x) * -Gc)
                loss_conv = F.mse_loss(g_S_masked, f_T_conv.detach())
                loss_pool = F.mse_loss(g_S_pooled, f_T_pooled.detach())
                
                loss = loss_conv + loss_pool
                
                loss.backward()
                optimizer_CKT.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.retain_loader)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time
            
            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | CKT MSE Loss: {avg_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)
            
            wandb.log({
                "epoch": epoch+1, 
                "ckt_loss": avg_loss, 
                "ra": ra_score, 
                "fa": fa_score, 
                "ta": ta_score, 
                "mia": mia_score
            })
            
            torch.save(model_S.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")
            
            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break
                
        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({"total_train_time_sec": total_unlearn_time, "peak_memory_gb": peak_memory_gb})

        torch.save(model_S.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time