from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_matthews_corrcoef, binary_f1_score, binary_precision, binary_recall, binary_average_precision
from data_utils import PrecomputedUniprotDataset
from focal_loss import WeightedFocalLoss
from model import ActiveSitePredictor, ActiveSitePredictorHead, IdentityNeck, AttentionNeck
import numpy as np
import os
import datetime
import wandb


def train_one_epoch(model: nn.Module, criterion: nn.Module, optimizer: torch.optim, train_loader: DataLoader, device: torch.device, last_n_layers: int = 0):
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

        optimizer.zero_grad()
        logits = model(embeddings, mask, last_n_layers=last_n_layers)
        torch_mask = (mask == 0)
        loss = criterion(logits, labels, torch_mask)
        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        all_masks.append((torch_mask == 0).cpu())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"\r  Train batch {batch_idx + 1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    results = compute_metrics(all_logits, all_labels, padding_mask=all_masks)
    avg_loss = total_loss / num_batches

    return avg_loss, results


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor = None, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    targets = targets.long()

    if padding_mask is not None:
        valid = ~padding_mask
        probs = probs[valid]
        targets = targets[valid]

    preds = (probs >= threshold).long()

    return {
        "mcc": binary_matthews_corrcoef(preds, targets).item(),
        "f1": binary_f1_score(preds, targets).item(),
        "precision": binary_precision(preds, targets).item(),
        "recall": binary_recall(preds, targets).item(),
        "auprc": binary_average_precision(probs, targets).item(),
    }


def find_optimal_threshold(all_logits: torch.Tensor, all_labels: torch.Tensor, padding_mask: torch.Tensor = None, num_thresholds: int = 100) -> tuple[float, float]:
    probs = torch.sigmoid(all_logits)
    targets = all_labels.long()

    if padding_mask is not None:
        valid = ~padding_mask
        probs = probs[valid]
        targets = targets[valid]

    thresholds = torch.linspace(0.01, 0.99, num_thresholds)
    best_mcc = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (probs >= thresh).long()

        if preds.sum() == 0 or (1 - preds).sum() == 0:
            continue

        mcc = binary_matthews_corrcoef(preds, targets).item()
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh.item()

    return best_thresh, best_mcc


def evaluate(model, loader, criterion, device, last_n_layers: int = 0, threshold: float = None):
    """
    Evaluate model on a dataloader.
    If threshold is None, find the optimal threshold on this data (use for val).
    If threshold is provided, use it directly (use for test, with val threshold).
    """
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

            torch_mask = (mask == 0)
            logits = model(embeddings, mask, last_n_layers=last_n_layers)
            loss = criterion(logits, labels, torch_mask)
            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(torch_mask.cpu())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"\r  Eval batch {batch_idx + 1}/{num_batches} | loss: {loss.item():.4f}", end="", flush=True)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    if threshold is None:
        # Find optimal threshold on this data (validation)
        best_threshold, _ = find_optimal_threshold(all_logits, all_labels, all_masks)
    else:
        # Use provided threshold (test — threshold was fixed on val)
        best_threshold = threshold

    results = compute_metrics(all_logits, all_labels, padding_mask=all_masks, threshold=best_threshold)
    results["threshold"] = best_threshold
    avg_loss = total_loss / num_batches

    return avg_loss, results


def print_final_summary(all_fold_val_results, all_fold_test_results, output_file="final_summary.txt"):
    metrics_to_report = ["mcc", "f1", "precision", "recall", "auprc"]

    with open(output_file, 'w') as f:
        for split_name, all_results in [("VAL", all_fold_val_results), ("TEST", all_fold_test_results)]:
            header = f"\n{'#' * 20} FINAL 5-FOLD {split_name} RESULTS {'#' * 20}\n"
            print(header)
            f.write(header + "\n")

            for m in metrics_to_report:
                values = [res[m] for res in all_results]
                values = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values]
                mean = np.mean(values)
                std = np.std(values)
                line = f"  {m:12}: {mean:.4f} +/- {std:.4f}"
                print(line)
                f.write(line + "\n")

            thresh_header = "\nOptimal thresholds per fold:"
            print(thresh_header)
            f.write(thresh_header + "\n")

            for fold_idx, res in enumerate(all_results):
                line = f"  Fold {fold_idx}: threshold={res['threshold']:.3f}"
                print(line)
                f.write(line + "\n")

    print(f"\nSummary saved to {output_file}")


def build_neck(neck_type: str, hidden_dim: int = 1280, dropout: float = None) -> nn.Module:
    valid = ["identity", "attention"]
    if neck_type not in valid:
        raise ValueError(f"Unknown neck type: '{neck_type}'. Valid options: {valid}")

    if neck_type == "identity":
        return IdentityNeck(output_dim=hidden_dim)
    elif neck_type == "attention":
        return AttentionNeck(hidden_dim=hidden_dim, n_layers=1, n_head=8, dropout=dropout)


def make_fold_splits(fold: int, all_folds: list[int]) -> tuple[list[int], int, int]:
    """
    For a given test fold index, return (train_folds, val_fold, test_fold).
    """
    test_fold = all_folds[fold]
    val_fold = all_folds[(fold - 1) % len(all_folds)]
    train_folds = [f for f in all_folds if f != test_fold and f != val_fold]
    return train_folds, val_fold, test_fold


def train_all_folds(device, neck_type: str = "identity", batch_size=32, warmup_epochs=5, total_epochs=80, lr=1e-3, weight_decay=0.01, neck_dropout: float = None, head_hidden_dim: int = 512,
                    precomputed_root="./data/train_val/precomputed_embeddings"):
    all_folds = [0, 1, 2, 3, 4]
    all_fold_val_results = []
    all_fold_test_results = []

    parent_run_dir = f"results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(parent_run_dir, exist_ok=True)

    model_dir = os.path.join(parent_run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    for fold in all_folds:
        train_folds, val_fold, test_fold = make_fold_splits(fold, all_folds)

        print(f"\n{'=' * 20} Run {fold} | train={train_folds} val=[{val_fold}] test=[{test_fold}] | neck={neck_type} {'=' * 20}")

        train_dataset = PrecomputedUniprotDataset(fold=train_folds, root=precomputed_root)
        val_dataset   = PrecomputedUniprotDataset(fold=[val_fold],   root=precomputed_root)
        test_dataset  = PrecomputedUniprotDataset(fold=[test_fold],  root=precomputed_root)

        neck = build_neck(neck_type, dropout=neck_dropout)
        last_n_layers = train_dataset.last_n_layers

        if last_n_layers > 0:
            model = ActiveSitePredictor(neck=neck, head_hidden_dim=head_hidden_dim)
            model.unfreeze_last_n_layers(last_n_layers)
        else:
            model = ActiveSitePredictorHead(neck=neck, head_hidden_dim=head_hidden_dim)

        model.to(device)

        if last_n_layers > 0:
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params = list(model.neck.parameters()) + list(model.classification_head.parameters())
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": lr * 0.01},
                {"params": head_params, "lr": lr},
            ], weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        n_positive = np.array(train_dataset.labels).sum()
        n_total = np.array(train_dataset.masks).sum()
        assert n_total > 0, "Total number of residues in training set is 0"
        pos_ratio = float(n_positive) / float(n_total)
        alpha = torch.tensor(1.0 - pos_ratio, dtype=torch.float32)
        criterion = WeightedFocalLoss(alpha=alpha, gamma=2.0)
        print(f"  pos_ratio={pos_ratio:.4f}, alpha={alpha:.4f}")

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])

        wandb.init(
            project="active-site-prediction",
            dir=parent_run_dir,
            group=neck_type,
            name=f"{neck_type}_run_{fold}_test{test_fold}",
            job_type="cross-validation",
            reinit=True,
            mode="offline"
        )
        wandb.config.update({
            "neck_type": neck_type,
            "last_n_layers": last_n_layers,
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "gamma": 2.0,
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "train_folds": train_folds,
            "val_fold": val_fold,
            "test_fold": test_fold,
        })

        best_val_mcc = -1.0
        best_val_results = None
        best_checkpoint_path = os.path.join(model_dir, f"best_model_run_{fold}.pt")

        for epoch in range(total_epochs):
            print(f"Epoch {epoch + 1}/{total_epochs}")
            train_loss, train_results = train_one_epoch(model, criterion, optimizer, train_loader, device, last_n_layers=last_n_layers)
            scheduler.step()

            val_loss, val_results = evaluate(model, val_loader, criterion, device, last_n_layers=last_n_layers, threshold=None)

            print(f"train_loss={train_loss:.4f} train_mcc={train_results['mcc']:.4f} | "
                  f"val_loss={val_loss:.4f} val_mcc={val_results['mcc']:.4f} "
                  f"val_auprc={val_results['auprc']:.4f} thresh={val_results['threshold']:.3f}")

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/mcc": train_results["mcc"],
                "train/f1": train_results["f1"],
                "train/precision": train_results["precision"],
                "train/recall": train_results["recall"],
                "val/loss": val_loss,
                "val/mcc": val_results["mcc"],
                "val/f1": val_results["f1"],
                "val/precision": val_results["precision"],
                "val/recall": val_results["recall"],
                "val/auprc": val_results["auprc"],
                "val/threshold": val_results["threshold"],
            })

            if val_results["mcc"] > best_val_mcc:
                best_val_mcc = val_results["mcc"]
                best_val_results = val_results
                print(f"  New best val MCC: {best_val_mcc:.4f} — saving checkpoint")

                torch.save({
                    "epoch": epoch,
                    "neck_type": neck_type,
                    "neck_dropout": neck_dropout,
                    "head_hidden_dim": head_hidden_dim,
                    "model": model.state_dict(),
                    "warmup_epochs": warmup_epochs,
                    "total_epochs": total_epochs,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "val_metrics": val_results,
                    "val_threshold": val_results["threshold"],
                    "train_folds": train_folds,
                    "val_fold": val_fold,
                    "test_fold": test_fold,
                }, best_checkpoint_path)

        # Load best checkpoint and evaluate on test using the val threshold
        print(f"\n  Loading best checkpoint for test evaluation (val MCC={best_val_mcc:.4f})...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        val_threshold = checkpoint["val_threshold"]

        test_loss, test_results = evaluate(model, test_loader, criterion, device, last_n_layers=last_n_layers, threshold=val_threshold)

        print(f"  Test MCC={test_results['mcc']:.4f} | F1={test_results['f1']:.4f} | "
              f"AUPRC={test_results['auprc']:.4f} | threshold={val_threshold:.3f}")

        wandb.log({
            "test/mcc": test_results["mcc"],
            "test/f1": test_results["f1"],
            "test/precision": test_results["precision"],
            "test/recall": test_results["recall"],
            "test/auprc": test_results["auprc"],
            "test/threshold": test_results["threshold"],
            "test/loss": test_loss,
        })

        all_fold_val_results.append(best_val_results)
        all_fold_test_results.append(test_results)
        wandb.finish()

    print_final_summary(all_fold_val_results, all_fold_test_results, output_file=os.path.join(parent_run_dir, f"final_summary_{neck_type}.txt"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Using device: {device}")

    train_all_folds(device, neck_type="identity", warmup_epochs=5, total_epochs=80, head_hidden_dim=512, precomputed_root="./data/uniprot/precomputed_embeddings")


if __name__ == "__main__":
    main()