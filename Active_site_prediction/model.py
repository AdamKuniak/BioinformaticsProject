from transformers import EsmModel
import torch
import torch.nn as nn

class ActiveSitePredictor(torch.nn.Module):
    def __init__(self, neck, model_name="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        self.backbone = EsmModel.from_pretrained(model_name)
        self.neck = neck
        self.classification_head = ClassificationHead(embedding_size=self.backbone.config.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Frozen backbone
        with torch.no_grad():
            outputs = self.backbone(input_ids=x, attention_mask=mask)
            embeddings = outputs.last_hidden_state
            # embeddings = outputs.last_hidden_state[:, 1:-1, :]  # strip <cls> and <eos>

        # Neck
        neck_output = self.neck(embeddings, mask)

        # Classification head
        return self.classification_head(neck_output)


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
