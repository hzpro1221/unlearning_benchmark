import torch

def forget_acc(model, forget_loader, device="cuda"):
    """
    calculates the accuracy of the model on a given dataloader.
    
    args:
        model: architecture inherited from BaseArchitecture.
        forget_loader: DataLoader containing the data to evaluate.
        device: "cuda" or "cpu".
        
    returns:
        float: accuracy as a ratio (0.0 to 1.0).
    """
    correct = 0
    total = 0
    
    for batch in forget_loader:
        # extract images and true labels
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # inference automatically handles eval() and no_grad()
        logits, _ = model.inference(images)
        
        predictions = torch.argmax(logits, dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    return correct / total