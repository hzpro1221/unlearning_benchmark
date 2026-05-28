import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

def mia(model, forget_loader, unseen_loader, device="cuda"):
    """
    Performs Membership Inference Attack via logistic regression on loss values.
    Ideal unlearned model yields score ~0.5 (forget set indistinguishable from unseen).
    """
    # reduction="none" gets individual loss per image
    criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_losses(loader):
        all_losses = []
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            logits, _ = model.inference(images)
            losses = criterion(logits, labels).cpu().detach().numpy()
            all_losses.extend(losses)
            
        return np.array(all_losses)

    forget_losses = compute_losses(forget_loader)
    unseen_losses = compute_losses(unseen_loader)

    # Balance datasets to prevent majority-class bias
    min_len = min(len(forget_losses), len(unseen_losses))
    if min_len == 0:
        raise ValueError("Length of forget set or unseen set is 0")
        
    forget_losses = forget_losses[:min_len]
    unseen_losses = unseen_losses[:min_len]

    # sklearn expects 2D arrays for features
    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape(-1, 1)
    
    # Labels -> 0: unseen (non-member), 1: forget (member)
    labels_mia = [0] * min_len + [1] * min_len

    attack_model = LogisticRegression()
    cv = StratifiedShuffleSplit(n_splits=10)
    
    mia_scores = cross_val_score(
        attack_model, samples_mia, labels_mia, cv=cv, scoring="accuracy"
    )
    
    return mia_scores.mean()