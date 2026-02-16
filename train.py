from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import torchmetrics
from torchmetrics import MetricCollection, MatthewsCorrCoef, JaccardIndex, F1Score
from torchmetrics.classification import MultilabelExactMatch
from data_utils import PrecomputedSwissDataset
from focal_loss import MultiLabelFocalLoss
from model import ProteinLocalizatorHead
import numpy as np


def train_one_epoch(model: nn.Module, criterion: nn.Module, optimizer: torch.optim, train_loader: DataLoader, train_metrics: MetricCollection, device: torch.device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        embedding = batch["embedding"].to(device, dtype=torch.float32)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # clear gradients
        optimizer.zero_grad()
        # forward pass
        logits, _ = model(embedding, mask)
        # Loss
        loss = criterion(logits, labels)
        # backward pas
        loss.backward()
        optimizer.step()
        # update metrics
        preds = torch.sigmoid(logits).squeeze(1)
        train_metrics.update(preds, labels.long())
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"\r  Train batch {batch_idx + 1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)
            print()

    results = train_metrics.compute()
    avg_loss = total_loss / len(train_loader)
    train_metrics.reset()

    return avg_loss, results

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, metrics: MetricCollection, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    num_batches = len(loader)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            embedding = batch["embedding"].to(device, dtype=torch.float32)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # forward pass
            logits, _ = model(embedding, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits)

            metrics.update(preds, labels.long())

            all_preds.append(preds.cpu())  # store on the cpu to save memory
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"\r  Eval batch {batch_idx + 1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)
                print()

        results = metrics.compute()
        avg_loss = total_loss / len(loader)
        metrics.reset()

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        opt_thresholds = find_optimal_thresholds(all_preds, all_labels)

        binary_preds = (all_preds >= opt_thresholds).long()
        mcc_per_class = [
            torchmetrics.functional.matthews_corrcoef(binary_preds[:, c], all_labels[:, c].long(), task="binary").item()
            for c in range(all_labels.shape[1])
        ]

        results["dev_mcc_macro"] = np.mean(mcc_per_class)  # Overwrite with the better score
        results["dev_mcc_per_class"] = mcc_per_class  # MCC per class
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
    # calculate mean and std for every metric
    for m in metrics_to_report:
        values = [res[m] for res in all_fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{m}: {mean:.4f} +/- {std:.4f}")

    # Per-Class Summary
    compartments = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

    print("\nPer-Class MCC (mean +/- std):")
    # all_fold_results[fold]["dev_mcc_per_class"] is a list of 10
    for i, name in enumerate(compartments):
        class_scores = [res["dev_mcc_per_class"][i] for res in all_fold_results]
        print(f"  {name:25}: {np.mean(class_scores):.4f} +/- {np.std(class_scores):.4f}")

def test_all_splits(metrics, device, batch_size=256, warmup_epochs=5, total_epochs=25, lr=0.00005, weight_decay=0.01):
    partitions = [0, 1, 2, 3, 4]
    all_fold_results = []

    for p in partitions:
        print(f"\n{'='*20} Starting fold {p} {'='*20}")

        # Initialize model (lightweight head only, no backbone on GPU)
        model = ProteinLocalizatorHead()
        model.to(device)

        # 2. Correct Partition Slicing
        train_folds = [f for f in partitions if f != p]
        dev_folds = [p]

        # Create datasets from precomputed embeddings
        train_dataset = PrecomputedSwissDataset(train_folds)
        dev_dataset = PrecomputedSwissDataset(dev_folds)

        # Get the class weights by extracting inverse frequencies
        counts = train_dataset.labels.sum(axis=0)
        total = len(train_dataset)
        inverse_frequencies = total / counts
        class_weights = torch.tensor(inverse_frequencies, dtype=torch.float).to(device)
        # Normalize
        class_weights = class_weights / class_weights.mean()

        # criterion
        criterion = MultiLabelFocalLoss(alphas=class_weights).to(device)

        # Wrap them with dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # warmup scheduler for easy start
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])

        train_metrics = metrics.clone(prefix="train_").to(device)
        dev_metrics = metrics.clone(prefix="dev_").to(device)

        compartments = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

        # Set up wandb
        wandb.init(group="Initial-Benchmark", name=f"fold_{p}", job_type="cross-validation", reinit=True, mode="offline")
        config = wandb.config
        config.learning_rate = lr
        config.batch_size = batch_size
        config.weight_decay = weight_decay

        best_mcc_this_fold = -1
        best_results_this_fold = None

        # training loop
        for i in range(total_epochs):
            train_loss, train_results = train_one_epoch(model, criterion, optimizer, train_loader, train_metrics, device)
            print(f"Epoch: {i+1}/{total_epochs}: Train Loss: {train_loss:.4f}, MCC: {train_results['train_mcc']}")
            scheduler.step()

            # print_metrics(i, total_epochs, train_loss, train_metrics, dataset="train")
            dev_loss, dev_results = evaluate(model, dev_loader, criterion, dev_metrics, device)

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

            if dev_results["dev_mcc"] > best_mcc_this_fold:
                best_mcc_this_fold = dev_results["dev_mcc"]
                best_results_this_fold = dev_results
                print(f"New Best Val MCC: {best_mcc_this_fold:.4f}")

                torch.save({
                    'epoch': i,
                    'attention_pooling': model.attention_pooling.state_dict(),
                    'classifier': model.classifier.state_dict(),
                    'metrics': dev_results,
                    'thresholds': dev_results['dev_thresholds']
                }, f"best_model_fold_{p}.pt")

        all_fold_results.append(best_results_this_fold)
        wandb.finish()

    print_final_summary(all_fold_results)


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set a seed for reproducibility
    torch.manual_seed(42)

    # Metrics to calculate; accuracy and mcc is per label
    metrics = MetricCollection({
        "exact_match_acc": MultilabelExactMatch(num_labels=10),
        "jaccard": JaccardIndex(task="multilabel", num_labels=10),
        "mcc": MatthewsCorrCoef(task="multilabel", num_labels=10),
        "f1_macro": F1Score(task="multilabel", num_labels=10, average="macro")
    })

    # Test all splits (run precompute.py first!)
    test_all_splits(metrics, device)


if __name__ == '__main__':
    main()
