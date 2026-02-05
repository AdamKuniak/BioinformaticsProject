import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alphas=None, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alphas", alphas)

    def forward(self, logits, targets):
        # shape[batch, num_classes]
        # reduction="none", I want to average after I add focal factor and weights
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # -log(p) for every class, where p is the probability of label being correct
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        # Calculate the focal factor
        focal_factor = (1 - pt) ** self.gamma
        # Focal loss
        loss = focal_factor * bce_loss
        # Weighted focal loss
        if self.alphas is not None:
            loss = loss * self.alphas

        # Average between batches and classes after focal factor, etc.
        return loss.mean()
