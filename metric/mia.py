import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

def mia(model, forget_loader, unseen_loader, device="cuda"):
    model.eval() 

    def compute_logits(loader):
        all_logits = []
        with torch.no_grad(): 
            for batch in loader:
                images = batch[0].to(device)
                
                # Extract raw prediction vectors 
                logits, _ = model.inference(images)
                all_logits.append(logits.cpu().numpy())
                
        return np.concatenate(all_logits, axis=0)

    forget_logits = compute_logits(forget_loader)
    unseen_logits = compute_logits(unseen_loader)

    # Balance datasets to prevent the attack model from exploiting a majority-class baseline
    min_len = min(len(forget_logits), len(unseen_logits))
    if min_len == 0:
        raise ValueError("Length of forget set or unseen set is 0")
        
    forget_logits = forget_logits[:min_len]
    unseen_logits = unseen_logits[:min_len]

    samples_mia = np.concatenate((unseen_logits, forget_logits), axis=0)
    
    # Target variables: 0 for non-members (unseen), 1 for members (forget)
    labels_mia = [0] * min_len + [1] * min_len

    attack_model = LogisticRegression(max_iter=1000) 
    cv = StratifiedShuffleSplit(n_splits=10, random_state=42) 
    
    mia_scores = cross_val_score(
        attack_model, samples_mia, labels_mia, cv=cv, scoring="accuracy"
    )
    
    return mia_scores.mean()