from transformers import EsmModel
import torch
import torch.nn as nn

class ActiveSitePredictor(nn.Module):
    """
    Main model class for active site prediction, which consists of a frozen ESM-2 backbone, a neck, and a classification head that produces binary predictions for each residue.
    """
    def __init__(self, neck, model_name="facebook/esm2_t33_650M_UR50D", head_hidden_dim=256):
        super().__init__()
        self.backbone = EsmModel.from_pretrained(model_name)
        self.neck = neck
        self.classification_head = ClassificationHead(
            embedding_size=self.neck.output_dim,
            hidden_dim=head_hidden_dim
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Frozen backbone
        with torch.no_grad():
            outputs = self.backbone(input_ids=x, attention_mask=mask)
            embeddings = outputs.last_hidden_state
            # embeddings = outputs.last_hidden_state[:, 1:-1, :]  # strip <cls> and <eos>

        # padding mask for the necks
        padding_mask = (mask == 0)

        # Neck
        neck_output = self.neck(embeddings, padding_mask)

        # Classification head
        return self.classification_head(neck_output)


class ClassificationHead(nn.Module):
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
        return self.classifier(x).squeeze(-1)


class IdentityNeck(nn.Module):
    """
    Identity neck, just passes backbone embeddings to the classification head without modification.
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x, mask=None):
        return x


class AttentionNeck(nn.Module):
    """
    Attention neck that applies self-attention to the backbone embeddings.
    """
    def __init__(self, hidden_dim: int, n_layers: int = 1, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.output_dim = hidden_dim

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,  # matches ESM-2 embedding size
            nhead=n_head,
            dim_feedforward=512,  # dimension of the FFN in the transformer layer
            dropout=dropout,
            batch_first=True
        )

        # the transformer itself
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_layers
        )

    def forward(self, x, mask=None):
        return self.transformer(x, src_key_padding_mask=mask)

