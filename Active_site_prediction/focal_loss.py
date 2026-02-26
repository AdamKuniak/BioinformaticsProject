import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Class implementing weighted focal loss for binary classification
    alpha : weight for the positive class (active site)
    gamma : focusing parameter — higher = more focus on hard examples
    """
    def __init__(self, alpha, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # bce[i] = -[y * log(sigmoid(x)) + (1-y) * log(1 - sigmoid(x))]
        # reduction = "none" means we keep the loss for each element in the batch and sequence, because I multiply each position's loss by different focal weight
        bce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(2), targets, reduction="none")  # [batch, seq_len]

        # get the probabilities
        probs = torch.sigmoid(logits)

        pt = probs * targets + (1 - probs) * (1 - targets)

        # If target == 1 alpha_t = alpha, if target == 0 alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # focal weights
        focal_weights = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weights * bce_loss

        # padding case
        if mask is not None:
            loss = loss.masked_fill(mask, 0.0)
            n_valid = (~mask).sum()
            return loss.sum() / n_valid.clamp(min=1)

        return loss.mean()
