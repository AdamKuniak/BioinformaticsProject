from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.functional.classification import binary_matthews_corrcoef, binary_f1_score, binary_precision, binary_recall, binary_average_precision
import wandb
from data_utils import PrecomputedUniprotDataset
from focal_loss import WeightedFocalLoss
from model import ActiveSitePredictor
import numpy as np
import os
import datetime


def train_one_epoch(model: nn.Module, criterion: nn.Module, optimizer: torch.optim, train_loader: DataLoader, train_metrics: MetricCollection, device: torch.device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    all_logits = []
    all_labels = []
    all_masks = []

    for batch_idx, batch in enumerate(train_loader):
        embeddings = batch["embedding"].to(device, dtype=torch.float32)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device, dtype=torch.float32)

        # clear gradients
        optimizer.zero_grad()
        # forward pass
        logits = model(embeddings, mask)
        # Loss
        torch_mask = (mask == 0)  # Torch has a different convention for masking than Hugging Face
        loss = criterion(logits, labels, torch_mask)
        # backward pas
        loss.backward()
        optimizer.step()

        # Accumulate for epoch-level metrics
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        all_masks.append((torch_mask == 0).cpu())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"\r  Train batch {batch_idx+1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)

    # Compute epoch-level metrics over all accumulated predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    results = compute_metrics(all_logits, all_labels, padding_mask=all_masks)
    avg_loss = total_loss / num_batches

    return avg_loss, results


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor = None, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    targets = targets.long()

    # Flatten and remove padding
    if padding_mask is not None:
        valid = ~padding_mask  # True for valid positions
        probs = probs[valid]
        targets = targets[valid]

    preds = (probs >= threshold).long()

    return {
        "mcc":       binary_matthews_corrcoef(preds, targets).item(),
        "f1":        binary_f1_score(preds, targets).item(),
        "precision": binary_precision(preds, targets).item(),
        "recall":    binary_recall(preds, targets).item(),
        "auprc":     binary_average_precision(probs, targets).item(),
    }


def find_optimal_threshold(all_logits: torch.Tensor, all_labels: torch.Tensor, padding_mask: torch.Tensor = None, num_thresholds: int = 100) -> tuple[float, float]:
    probs = torch.sigmoid(all_logits)
    targets = all_labels.long()

    # Flatten and remove padding before searching
    if padding_mask is not None:
        valid = ~padding_mask
        probs = probs[valid]
        targets = targets[valid]

    thresholds = torch.linspace(0.01, 0.99, num_thresholds)
    best_mcc = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (probs >= thresh).long()

        # Skip degenerate cases where MCC is undefined
        if preds.sum() == 0 or (1 - preds).sum() == 0:
            continue

        mcc = binary_matthews_corrcoef(preds, targets).item()
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh.item()

    return best_thresh, best_mcc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    all_logits = []
    all_labels = []
    all_masks = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            embeddings = batch["embedding"].to(device, dtype=torch.float32)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device, dtype=torch.float32)

            torch_mask = (mask == 0)  # Torch has a different convention for masking than Hugging Face
            logits = model(embeddings, mask)
            loss = criterion(logits, labels, torch_mask)
            total_loss += loss.item()

            # Accumulate on CPU
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(torch_mask.cpu())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"\r  Eval batch {batch_idx+1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)

    # [n_proteins, seq_len]
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks,  dim=0)

    # Find the best threshold on this validation set
    best_threshold, best_mcc = find_optimal_threshold(all_logits, all_labels, all_masks)

    # Compute full metrics at the best threshold
    results = compute_metrics(all_logits, all_labels, padding_mask=all_masks, threshold=best_threshold)

    results["threshold"] = best_threshold
    avg_loss = total_loss / num_batches

    return avg_loss, results


def print_final_summary(all_fold_results, output_file="final_summary.txt"):
    metrics_to_report = ["mcc", "f1", "precision", "recall", "auprc"]

    with open(output_file, 'w') as f:
        header = f"\n{'#' * 20} FINAL 5-FOLD CROSS-VALIDATION RESULTS {'#' * 20}\n"
        print(header)
        f.write(header + "\n")

        for m in metrics_to_report:
            values = [res[m] for res in all_fold_results]
            values = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values]
            mean = np.mean(values)
            std = np.std(values)
            line = f"  {m:12}: {mean:.4f} +/- {std:.4f}"
            print(line)
            f.write(line + "\n")

        # Report best threshold per fold
        thresh_header = "\nOptimal thresholds per fold:"
        print(thresh_header)
        f.write(thresh_header + "\n")

        for fold_idx, res in enumerate(all_fold_results):
            line = f"  Fold {fold_idx}: threshold={res['threshold']:.3f}"
            print(line)
            f.write(line + "\n")

    print(f"\nSummary saved to {output_file}")
