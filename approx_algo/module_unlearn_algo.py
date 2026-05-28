import torch
import torch.nn.functional as F

def module_unlearn_algo(model, origin_model, criteria, optimizer, forget_loader, retain_loader, 
                        selected_experts_per_layer, device="cuda", beta=1.0, gamma=1.0, eta=1.0):
    # unlearning optimization loop
    model.train()
    origin_model.eval()
    retain_iter = iter(retain_loader)
    total_loss_accum = 0.0
    
    moe_layers = [m for _, m in model.named_modules() if m.__class__.__name__ == 'DeepMoELayer']
    
    for forget_batch in forget_loader:
        images_f = forget_batch[0].to(device)
        labels_f = forget_batch[1].to(device)
        
        try:
            retain_batch = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            retain_batch = next(retain_iter)
            
        images_r = retain_batch[0].to(device)
        labels_r = retain_batch[1].to(device)
        
        optimizer.zero_grad()
        
        logits_f, _ = model.forward_with_grad(images_f)
        loss_forget = -criteria(logits_f, labels_f)
        
        logits_r, _ = model.forward_with_grad(images_r)
        loss_retain = criteria(logits_r, labels_r)
        
        with torch.no_grad():
            orig_logits_r, _ = origin_model.forward_with_grad(images_r)
            
        log_preds = F.log_softmax(logits_r, dim=-1)
        target_preds = F.softmax(orig_logits_r, dim=-1)
        loss_distill = F.kl_div(log_preds, target_preds, reduction='batchmean')
        
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

        total_loss = loss_forget + (beta * loss_retain) + (gamma * loss_distill) + (eta * loss_sep)
        
        total_loss.backward()
        optimizer.step()
        total_loss_accum += total_loss.item()

    return total_loss_accum / len(forget_loader)