from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.functional.classification import binary_matthews_corrcoef, binary_f1_score, binary_precision, binary_recall, binary_average_precision
from data_utils import PrecomputedUniprotDataset
from focal_loss import WeightedFocalLoss
from model import ActiveSitePredictor, IdentityNeck, AttentionNeck
import numpy as np
import os
import datetime
import wandb


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

        # Gradient clipping — important for transformer neck stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def build_neck(neck_type: str, hidden_dim: int = 1280) -> nn.Module:
    """
        Neck factory to build the neck module based on the specified type.
    """
    valid = ["identity", "attention"]
    if neck_type not in valid:
        raise ValueError(f"Unknown neck type: '{neck_type}'. "
                         f"Valid options: {valid}")

    if neck_type == "identity":
        return IdentityNeck(output_dim=hidden_dim)
    elif neck_type == "attention":
        return AttentionNeck(hidden_dim=hidden_dim, n_layers=1, n_head=8, dropout=0.1)


def train_all_folds(device, neck_type: str = "identity", batch_size=32, warmup_epochs=3, total_epochs=20, lr=1e-3, weight_decay=0.01):
    partitions = [0, 1, 2, 3, 4]
    all_fold_results = []

    parent_run_dir = f"results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(parent_run_dir, exist_ok=True)

    model_dir = os.path.join(parent_run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    for p in partitions:
        print(f"\n{'=' * 20} Fold {p} | neck={neck_type} {'=' * 20}")

        # Build a fresh model for each fold
        neck = build_neck(neck_type)
        model = ActiveSitePredictor(neck=neck, head_hidden_dim=512)
        model.to(device)

        # Datasets
        train_folds = [f for f in partitions if f != p]
        train_dataset = PrecomputedUniprotDataset(fold=train_folds)
        dev_dataset = PrecomputedUniprotDataset(fold=[p])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Criterion
        n_positive = train_dataset.labels.sum()
        n_total = (train_dataset.masks == 1).sum()
        pos_ratio = float(n_positive) / float(n_total)
        alpha = 1.0 - pos_ratio
        criterion = WeightedFocalLoss(alpha=alpha, gamma=2.0)
        print(f"  pos_ratio={pos_ratio:.4f}, alpha={alpha:.4f}")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])

        wandb.init(
            project="active-site-prediction",
            dir=parent_run_dir,
            group=neck_type,  # group runs by architecture
            name=f"{neck_type}_fold_{p}",
            job_type="cross-validation",
            reinit=True,
            mode="offline"
        )
        wandb.config.update({
            "neck_type": neck_type,
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "gamma": 2.0,
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
        })

        best_mcc_this_fold = -1.0
        best_results_this_fold = None

        for i in range(total_epochs):
            train_loss, train_results = train_one_epoch(model, criterion, optimizer, train_loader, device)
            scheduler.step()

            dev_loss, dev_results = evaluate(model, dev_loader, criterion, device)

            print(f"Epoch {i + 1}/{total_epochs} | "
                  f"train_loss={train_loss:.4f} train_mcc={train_results['mcc']:.4f} | "
                  f"dev_loss={dev_loss:.4f} dev_mcc={dev_results['mcc']:.4f} "
                  f"dev_auprc={dev_results['auprc']:.4f} "
                  f"thresh={dev_results['threshold']:.3f}")

            wandb.log({
                "epoch": i + 1,
                "train/loss": train_loss,
                "train/mcc": train_results["mcc"],
                "train/f1": train_results["f1"],
                "train/precision": train_results["precision"],
                "train/recall": train_results["recall"],
                "dev/loss": dev_loss,
                "dev/mcc": dev_results["mcc"],
                "dev/f1": dev_results["f1"],
                "dev/precision": dev_results["precision"],
                "dev/recall": dev_results["recall"],
                "dev/auprc": dev_results["auprc"],
                "dev/threshold": dev_results["threshold"],
            })

            if dev_results["mcc"] > best_mcc_this_fold:
                best_mcc_this_fold = dev_results["mcc"]
                best_results_this_fold = dev_results
                print(f"  New best MCC: {best_mcc_this_fold:.4f}")

                torch.save({
                    "epoch": i,
                    "neck_type": neck_type,
                    "model": model.state_dict(),
                    "metrics": dev_results,
                    "threshold": dev_results["threshold"],
                }, os.path.join(model_dir, f"best_model_fold_{p}.pt"))

        all_fold_results.append(best_results_this_fold)
        wandb.finish()

    print_final_summary(all_fold_results, output_file=os.path.join(parent_run_dir, f"final_summary_{neck_type}.txt"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)  # for numpy reproducibility too

    print(f"Using device: {device}")

    train_all_folds(device, neck_type="identity")


if __name__ == "__main__":
    main()
