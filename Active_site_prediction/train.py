from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import torchmetrics
from torchmetrics import MetricCollection, MatthewsCorrCoef, JaccardIndex, F1Score
from torchmetrics.classification import MultilabelExactMatch
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
        # update metrics
        preds = torch.sigmoid(logits).squeeze(1)
        train_metrics.update(preds, labels.long())
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"\r  Train batch {batch_idx+1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)
            print()

    results = train_metrics.compute()
    avg_loss = total_loss / len(train_loader)
    train_metrics.reset()

    return avg_loss, results
