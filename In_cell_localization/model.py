from transformers import EsmModel
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
    Class implementing the attention pooling, which takes the protein embedding as input and produces a pooled representation of the protein using attention mechanism
    hidden_size: size of the embedding dimension
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.learnable_vector = nn.Parameter(torch.empty(1, embedding_size))
        self.learnable_vector = nn.init.xavier_uniform_(self.learnable_vector)

        # 1D Gaussian filter
        kernel_size = 5
        std = 1
        x = torch.arange(kernel_size) - (kernel_size - 1) // 2  # array [-2, -1, 0, 1, 2]
        g_filter = torch.exp(-(x.pow(2)) / (2 * std ** 2))  # Gaussian filter
        g_filter = g_filter.view(1, 1, -1)  # conv1d expects dimensions [in_channels, out_channels, width]
        self.register_buffer("gaussian_filter", g_filter)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert embeddings.shape[2] == self.learnable_vector.T.shape[0], f"Dimension mismatch: {embeddings.shape[2]} != {self.learnable_vector.T.shape[0]}"
        raw_scores = embeddings @ self.learnable_vector.T.squeeze(-1)

        smoothed_scores = nn.functional.conv1d(raw_scores.unsqueeze(1), self.gaussian_filter, padding=2)
        smoothed_scores = smoothed_scores.squeeze(1)  # [batch, seq_len]
        smoothed_scores = smoothed_scores.masked_fill(mask == 0, value=-1e9)

        # Softmax to get attention weights
        attention_weights = nn.functional.softmax(smoothed_scores, dim=-1)
        assert attention_weights.shape[1] == embeddings.shape[1], f"Dimension mismatch: {attention_weights.shape[1]} != {embeddings.shape[1]}"
        # The weighted sum
        final_representation = torch.bmm(attention_weights.unsqueeze(1), embeddings)

        return final_representation, attention_weights


class ProteinLocalizatorHead(nn.Module):
    """
    Lightweight head, just attention pooling & classifier that operates on precomputed ESM-2 embeddings.
    No backbone on GPU — use with PrecomputedDataset for fast training.
    """
    def __init__(self, embedding_size=1280, num_labels=10):
        super().__init__()
        self.hidden_layer_size = 128
        self.dropout_prob = 0.1

        self.attention_pooling = AttentionPooling(embedding_size)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=self.hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features=self.hidden_layer_size, out_features=num_labels)
        )

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_embedding, att_weights = self.attention_pooling(embeddings, attention_mask)
        logits = self.classifier(pooled_embedding.squeeze(1))
        return logits, att_weights


class ProteinLocalizator(nn.Module):
    """
    Full model with ESM-2 backbone + classification head.
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", num_labels=10):
        super().__init__()
        # Pretrained backbone, ESM-2 model
        self.backbone = EsmModel.from_pretrained(model_name)

        self.head = ProteinLocalizatorHead(
            embedding_size=self.backbone.config.hidden_size,
            num_labels=num_labels
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Frozen backbone
        with torch.no_grad():
            outputs = self.backbone(input_ids=x, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # Shape: [Batch, Seq, 1280]

        # Head
        logits = self.head(embeddings, attention_mask)

        return logits
