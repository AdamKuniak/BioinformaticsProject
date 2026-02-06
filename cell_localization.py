from torch.utils.data import DataLoader
from transformers import EsmModel, AutoTokenizer
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MatthewsCorrCoef, JaccardIndex, F1Score, Accuracy
from data_utils import ProteinLocalizationDataset
from focal_loss import MultiLabelFocalLoss

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

    def forward(self, embeddings : torch.Tensor, mask : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert embeddings.shape[2] == self.learnable_vector.T.shape[0], f"Dimension mismatch: {embeddings.shape[2]} != {self.learnable_vector.T.shape[0]}"
        raw_scores = embeddings @ self.learnable_vector.T.squeeze(-1)

        smoothed_scores = nn.functional.conv1d(raw_scores.unsqueeze(1), self.gaussian_filter, padding=2)
        smoothed_scores = smoothed_scores.squeeze(1)  # [batch, seq_len]
        smoothed_scores = smoothed_scores.masked_fill(mask=mask, value=-1e9)

        # Softmax to get attention weights
        attention_weights = nn.functional.softmax(smoothed_scores, dim=-1)
        assert attention_weights.shape[1] == embeddings.shape[1], f"Dimension mismatch: {attention_weights.shape[1]} != {embeddings.shape[1]}"
        # The weighted sum
        final_representation = torch.bmm(attention_weights.unsqueeze(1), embeddings)

        return final_representation, attention_weights


class ProteinLocalizator(nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", num_labels=10):
        super().__init__()
        # Pretrained backbone, ESM-2 model
        self.backbone = EsmModel.from_pretrained(model_name)
        self.embedding_size = self.backbone.config.hidden_size
        self.hidden_size = 128
        self.dropout_prob = 0.1

        self.attention_pooling = AttentionPooling(self.embedding_size)

        self.classifier_head = nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features=self.hidden_size, out_features=num_labels)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Frozen backbone
        with torch.inference_mode():
            outputs = self.backbone(input_ids=x, attention_mask=attention_mask)

            embeddings = outputs.last_hidden_state # Shape: [Batch, Seq, 1280]
        # Learnable parts
        pooled_embedding, att_weights = self.attention_pooling(embeddings, attention_mask)
        logits = self.classifier_head(pooled_embedding)

        return logits, att_weights

def train_one_epoch(model, criterion, optimizer, train_loader, train_metrics):
    model.train()
    model.backbone.eval()
    total_loss = 0.0

    for batch in train_loader:
        data = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["label"]
        # clear gradients
        optimizer.zero_grad()
        # forward pass
        logits, _ = model(data, mask)
        # Loss
        loss = criterion(logits, labels)
        # backward pas
        loss.backward()
        optimizer.step()
        # update metrics
        preds = torch.sigmoid(logits)
        train_metrics.update(preds, labels)
        total_loss += loss.item()

    results = train_metrics.compute()
    avg_loss = total_loss / len(train_loader)
    train_metrics.reset()

    return avg_loss, results

def evaluate(model, val_loader, criterion, val_metrics):
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            data = batch["input_ids"]
            mask = batch["attention_mask"]
            labels = batch["label"]

            logits, _ = model(data, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(logits)
            val_metrics.update(preds, labels)

        results = val_metrics.compute()
        avg_loss = total_loss / len(val_loader)
        val_metrics.reset()

    return avg_loss, results

def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = ProteinLocalizator()
    # Freeze the ESM-2 backbone and set to eval, I want to train only the AttentionPooling and the classifier head
    model.backbone.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False

    train_split = [1, 2, 3, 4]
    val_split = [0]
    # Create datasets
    train_dataset = ProteinLocalizationDataset(tokenizer, train_split)
    val_dataset = ProteinLocalizationDataset(tokenizer, val_split)

    # Get the class weights by extracting inverse frequencies
    counts = train_dataset.labels.sum(axis=0)
    total = len(train_dataset)
    inverse_frequencies = total / counts
    class_weights = torch.tensor(inverse_frequencies, dtype=torch.float)
    # Normalize
    class_weights = class_weights / class_weights.mean()
    # criterion
    criterion = MultiLabelFocalLoss(alphas=class_weights)

    # Wrap them with dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    warmup_epochs = 5
    total_epochs = 25
    lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # warmup scheduler for easy start
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])

    # Metrics to calculate; accuracy and mcc is per label
    metrics = MetricCollection({
        "accuracy": Accuracy(task="multilabel", num_labels=10, subset_accuracy="True", average=None),
        "jaccard": JaccardIndex(task="multilabel", num_labels=10),
        "mcc": MatthewsCorrCoef(task="multilabel", num_labels=10, average=None),
        "f1_macro": F1Score(task="multilabel", num_labels=10, average="macro")
    })
    train_metrics = metrics.clone(prefix="train_")
    val_metrics = metrics.clone(prefix="val_")

    for i in range(total_epochs):
        avg_loss, results = train_one_epoch(model, criterion, optimizer, train_loader, train_metrics)
        print(f"Epoch: {i+1}/{total_epochs}\n"
              f"Train Loss: {avg_loss:.4f}\n"
              f"MCC: {results['mcc']:.4f}\n"
              f"Accuracy: {results['accuracy']:.4f}\n"
              f"F1: {results['f1_macro']:.4f}\n"
              f"Jaccard: {results['jaccard']:.4f}\n"
        )

if __name__ == '__main__':
    main()
