from transformers import EsmModel, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.learnable_vector = nn.Parameter(torch.empty(1, hidden_size))
        self.learnable_vector = nn.init.xavier_uniform_(self.learnable_vector)

        # 1D Gaussian filter
        kernel_size = 5
        std = 1
        x = torch.arange(kernel_size) - (kernel_size - 1) // 2  # array [-2, -1, 0, 1, 2]
        filter = torch.exp(-(x.pow(2)) / (2 * std ** 2))  # Gaussian filter
        filter = filter.view(1, 1, -1)  # conv1d expects dimensions [in_channels, out_channels, width]
        self.register_buffer("gaussian_filter", filter)

    def forward(self, embeddings : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        assert embeddings.shape[2] == self.learnable_vector.T.shape[0], f"Dimension mismatch: {embeddings.shape[2]} != {self.learnable_vector.T.shape[0]}"
        raw_scores = embeddings @ self.learnable_vector.T.squeeze(-1)

        smoothed_scores = nn.functional.conv1d(raw_scores.unsqueeze(1), self.gaussian_filter, padding=2)
        smoothed_scores = smoothed_scores.squeeze(1) # [batch, seq_len]
        smoothed_scores = smoothed_scores.masked_fill(mask=mask, value=-1e9)

        # Softmax to get attention weights
        attention_weights = nn.functional.softmax(smoothed_scores, dim=-1)
        assert attention_weights.shape[1] == embeddings.shape[1], f"Dimension mismatch: {attention_weights.shape[1]} != {embeddings.shape[1]}"
        # The weighted sum
        final_representation = torch.bmm(attention_weights.unsqueeze(1), embeddings)

        return final_representation


class ProteinLocalizator(nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", num_labels=10):
        super().__init__()
        # Pretrained backbone, ESM-2 model
        self.backbone = EsmModel.from_pretrained(model_name)
        self.embedding_size = self.backbone.config.hidden_size

def main():
    print("Working on it!")

if __name__ == '__main__':
    main()