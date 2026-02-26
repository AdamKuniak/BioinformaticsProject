from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.functional.classification import binary_matthews_corrcoef, binary_f1_score, binary_precision, binary_recall, binary_average_precision

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
        loss = criterion(logits, labels, mask == 0)
        # backward pas
        loss.backward()
        optimizer.step()

        # Accumulate for epoch-level metrics
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        all_masks.append((mask == 0).cpu())

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
