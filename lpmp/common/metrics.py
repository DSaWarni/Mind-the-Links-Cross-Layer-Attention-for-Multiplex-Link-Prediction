
import torch

@torch.no_grad()
def binary_scores_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Args:
        logits: (N, C) raw logits
        targets: (N, C) in {0,1}
    Returns:
        dict with 'acc', 'f1_macro', 'precision_macro', 'recall_macro', 'auc_approx'
    Note: We avoid adding sklearn to keep deps minimal; AUC is approximated with ranking-based method
    for a **single** output position (index 0) which is the target layer in this pipeline.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)
    correct = (preds == targets).float().mean().item()

    eps = 1e-8
    # macro over classes (columns)
    precisions, recalls, f1s = [], [], []
    for c in range(targets.size(1)):
        y = targets[:, c]
        p = preds[:, c]
        tp = (p*y).sum().float()
        fp = (p*(1-y)).sum().float()
        fn = ((1-p)*y).sum().float()
        prec = (tp / (tp + fp + eps)).item()
        rec  = (tp / (tp + fn + eps)).item()
        if prec + rec > 0:
            f1 = 2*prec*rec/(prec+rec)
        else:
            f1 = 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # Simple AUC approximation for column 0 (target layer) using rank statistic
    # This is deterministic given logits, no thresholding.
    try:
        s = probs[:, 0]
        y = targets[:, 0].float()
        pos = s[y==1]; neg = s[y==0]
        if pos.numel() > 0 and neg.numel() > 0:
            # U-statistic approximation
            auc = ((pos.view(-1,1) > neg.view(1,-1)).float().mean().item() +
                   0.5*((pos.view(-1,1) == neg.view(1,-1)).float().mean().item()))
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    return {
        "acc": correct,
        "precision_macro": sum(precisions)/len(precisions),
        "recall_macro": sum(recalls)/len(recalls),
        "f1_macro": sum(f1s)/len(f1s),
        "auc_approx": auc,
    }
