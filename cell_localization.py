from torch.utils.data import DataLoader
from transformers import EsmModel, AutoTokenizer
import torch
import torch.nn as nn
import wandb
import torchmetrics
from torchmetrics import MetricCollection, MatthewsCorrCoef, JaccardIndex, F1Score
from torchmetrics.classification import MultilabelExactMatch
from data_utils import ProteinLocalizationDataset
from focal_loss import MultiLabelFocalLoss
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
        with torch.no_grad():
            outputs = self.backbone(input_ids=x, attention_mask=attention_mask)

            embeddings = outputs.last_hidden_state # Shape: [Batch, Seq, 1280]
        # Learnable parts
        pooled_embedding, att_weights = self.attention_pooling(embeddings, attention_mask)
        logits = self.classifier_head(pooled_embedding)

        return logits, att_weights

def train_one_epoch(model: nn.Module, criterion: nn.Module, optimizer: torch.optim, train_loader: DataLoader, train_metrics: MetricCollection):
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
        preds = torch.sigmoid(logits).squeeze(1)
        train_metrics.update(preds, labels.long())
        total_loss += loss.item()

    results = train_metrics.compute()
    avg_loss = total_loss / len(train_loader)
    train_metrics.reset()

    return avg_loss, results

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, metrics: MetricCollection):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in loader:
            data = batch["input_ids"]
            mask = batch["attention_mask"]
            labels = batch["label"]

            # forward pass
            logits, _ = model(data, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits)

            metrics.update(preds, labels)

            all_preds.append(preds)
            all_labels.append(labels)

        results = metrics.compute()
        avg_loss = total_loss / len(loader)
        metrics.reset()

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        opt_thresholds = find_optimal_thresholds(all_preds, all_labels)

        binary_preds = (all_preds >= opt_thresholds).long()
        mcc_per_class = torchmetrics.functional.matthews_corrcoef(binary_preds, all_labels.long(), task="multilabel", num_labels=10)

        results["dev_mcc_macro"] = mcc_per_class.mean().item()  # Overwrite with the better score
        results["dev_mcc_per_class"] = mcc_per_class.tolist()  # MCC per class
        results["dev_thresholds"] = opt_thresholds

    return avg_loss, results

def find_optimal_thresholds(all_preds: torch.Tensor, all_labels: torch.Tensor, num_labels=10):
    optimal_thresholds = torch.zeros(num_labels)

    # Different thresholds to try for each class
    thresholds_range = torch.linspace(0.05, 0.95, 90)

    for class_idx in range(num_labels):
        best_mcc_for_class = -1.0
        best_threshold_for_class = 0.5  # Default

        # predictions and labels for the current class
        class_preds = all_preds[:, class_idx]
        class_labels = all_labels[:, class_idx]

        for threshold in thresholds_range:
            binary_preds = (class_preds >= threshold).long()

            # MCC for this class with given threshold
            # ensure at least one positive and one negative prediction and label, so MCC can be valid
            if binary_preds.sum() > 0 and (1 - binary_preds).sum() > 0 and class_labels.sum() > 0 and (1 - class_labels).sum() > 0:
                current_mcc = torchmetrics.functional.matthews_corrcoef(binary_preds, class_labels.long(), task="binary")

                # update the best threshold if higher MCC achieved
                if current_mcc > best_mcc_for_class:
                    best_mcc_for_class = current_mcc
                    best_threshold_for_class = threshold

        optimal_thresholds[class_idx] = best_threshold_for_class

    return optimal_thresholds


def print_final_summary(all_fold_results):
    metrics_to_report = ["dev_mcc_macro", "dev_exact_match_acc", "dev_f1_macro", "dev_jaccard"]

    print(f"\n{'#' * 20} FINAL 5-FOLD CROSS-VALIDATION RESULTS {'#' * 20}")
    for m in metrics_to_report:
        values = [res[m] for res in all_fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{m}: {mean:.4f} +/- {std:.4f}")

    # Per-Class Summary
    compartments = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

    print("\nPer-Class MCC (Mean +/- Std):")
    # all_fold_results[fold]["dev_mcc_per_class"] is a list of 10
    for i, name in enumerate(compartments):
        class_scores = [res["dev_mcc_per_class"][i] for res in all_fold_results]
        print(f"  {name:25}: {np.mean(class_scores):.4f} +/- {np.std(class_scores):.4f}")

def test_all_splits(tokenizer, metrics, batch_size=64, warmup_epochs=5, total_epochs=25, lr=0.001, weight_decay=0.01):
    partitions = [0, 1, 2, 3, 4]
    all_fold_results = []

    for p in partitions:
        print(f"\n{'='*20} STARTING FOLD {p} {'='*20}")

        # Initialize model
        model = ProteinLocalizator()
        # Freeze backbone
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 2. Correct Partition Slicing
        train_folds = [f for f in partitions if f != p]
        dev_folds = [p]

        # Create datasets
        train_dataset = ProteinLocalizationDataset(tokenizer, train_folds)
        dev_dataset = ProteinLocalizationDataset(tokenizer, dev_folds)

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # warmup scheduler for easy start
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])

        train_metrics = metrics.clone(prefix="train_")
        dev_metrics = metrics.clone(prefix="dev_")

        compartments = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

        # Set up wandb
        wandb.init(group="Initial-Benchmark", name=f"fold_{p}", reinit=True)
        config = wandb.config
        config.learning_rate = lr
        config.batch_size = batch_size
        config.weight_decay = weight_decay

        best_mcc_this_fold = -1
        best_results_this_fold = None
        # training loop
        for i in range(total_epochs):
            train_loss, train_results = train_one_epoch(model, criterion, optimizer, train_loader, train_metrics)
            scheduler.step()

            # print_metrics(i, total_epochs, train_loss, train_metrics, dataset="train")
            dev_loss, dev_results = evaluate(model, dev_loader, criterion, dev_metrics)

            # Wandb log
            metrics_to_log = {
                f"epoch": i + 1,
                "train/loss": train_loss,
                "train/mcc": train_results["train_mcc"],
                "train/exact_acc": train_results["train_exact_match_acc"],  # Use whatever key your MetricCollection outputs

                "dev/loss": dev_loss,
                "dev/mcc_macro": dev_results["dev_mcc"],
                "dev/exact_acc": dev_results["dev_exact_match_acc"],
                "dev/f1_macro": dev_results["dev_f1_macro"],
                "dev/jaccard": dev_results["dev_jaccard"]
            }

            # Per-Class MCCs to the log
            for name, score in zip(compartments, dev_results["dev_mcc_per_class"]):
                metrics_to_log[f"val_per_class_mcc/{name}"] = score

            wandb.log(metrics_to_log)

            if dev_results["dev_mcc_macro"] > best_mcc_this_fold:
                best_mcc_this_fold = dev_results["dev_mcc_macro"]
                best_results_this_fold = dev_results
                print(f"New Best Val MCC: {best_mcc_this_fold:.4f}")

                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': dev_results,
                    'thresholds': dev_results['dev_thresholds']
                }, f"best_model_fold_{p}.pt")

            all_fold_results.append(best_results_this_fold)
            wandb.finish()

        print_final_summary(all_fold_results)

def main():
    # set a seed for reproducibility
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Metrics to calculate; accuracy and mcc is per label
    metrics = MetricCollection({
        "exact_match_acc": MultilabelExactMatch(num_labels=10),
        "jaccard": JaccardIndex(task="multilabel", num_labels=10),
        "mcc": MatthewsCorrCoef(task="multilabel", num_labels=10),
        "f1_macro": F1Score(task="multilabel", num_labels=10, average="macro")
    })

    # Test all splits
    test_all_splits(tokenizer, metrics)


if __name__ == '__main__':
    main()
