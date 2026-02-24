from transformers import EsmModel
import torch
import torch.nn as nn


class ClassificationHead(torch.nn.Module):
    """
    Simple classification head that produces a binary prediction for each residue.
    """
    def __init__(self, embedding_size: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, x):
        return self.classifier(x).squeze(-1)
