import torch

def l1_sparse(model, criteria, optimizer, forget_loader, alpha=0.1, device="cuda"): # Có thể giảm nhẹ alpha xuống 0.1
    model.train()
    total_batch_loss = 0.0
    
    for batch in forget_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        optimizer.zero_grad()
        logits, _ = model.forward_with_grad(images)
        ce_loss = criteria(logits, labels)
        
        l1_penalty = 0.0
        num_params = 0 
        
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' not in name and 'norm' not in name:
                l1_penalty += torch.sum(torch.abs(param))
                num_params += param.numel() 
        
        if num_params > 0:
            l1_penalty = l1_penalty / num_params
        
        total_loss = -(ce_loss) + (alpha * l1_penalty)
        
        total_loss.backward()
        optimizer.step()
        
        total_batch_loss += total_loss.item()
        
    avg_loss = total_batch_loss / len(forget_loader)
    return avg_loss