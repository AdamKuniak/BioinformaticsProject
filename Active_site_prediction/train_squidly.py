from torch.utils.data import DataLoader, random_split
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

from train import (
    compute_metrics,
    find_optimal_threshold,
    train_one_epoch,
    evaluate,
    build_neck,
)


def print_final_summary(best_val_results, test_results, output_file="final_summary.txt"):
    metrics_to_report = ["mcc", "f1", "precision", "recall", "auprc"]

    with open(output_file, 'w') as f:
        for split_name, results in [("VAL (best MCC)", best_val_results),
                                    ("TEST (uni3175)", test_results)]:
            header = f"\n{'#' * 20} {split_name} {'#' * 20}\n"
            print(header)
            f.write(header + "\n")

            for m in metrics_to_report:
                v = results[m]
                v = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                line = f"  {m:12}: {v:.4f}"
                print(line)
                f.write(line + "\n")

            thresh_line = f"  {'threshold':12}: {results['threshold']:.3f}"
            print(thresh_line)
            f.write(thresh_line + "\n")

    print(f"\nSummary saved to {output_file}")


def train_squidly(
        device,
        neck_type="identity",
        batch_size=32,
        warmup_epochs=5,
        total_epochs=80,
        lr=1e-3,
        weight_decay=0.01,
        neck_dropout=0.1,
        head_hidden_dim=512,
        val_split=0.1,  # 90:10 split
        train_json="./data/squidly/uni14230_clean.json",
        test_json="./data/squidly/uni3175_clean.json",
        precomputed_root="./data/squidly/precomputed_embeddings",
):
    last_n_layers = 0  # set to >0 if using fine-tuning variant

    run_dir = f"results/squidly/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{neck_type}"
    os.makedirs(run_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ── Datasets ────────────────────────────────────────────────────────────
    full_train_dataset = PrecomputedUniprotDataset(root=precomputed_root)
    test_dataset = PrecomputedUniprotDataset(root=precomputed_root)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset,[train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    neck = build_neck(neck_type, dropout=neck_dropout)

    if last_n_layers > 0:
        model = ActiveSitePredictor(neck=neck, head_hidden_dim=head_hidden_dim)
        model.unfreeze_last_n_layers(last_n_layers)
    else:
        model = ActiveSitePredictorHead(neck=neck, head_hidden_dim=head_hidden_dim)

    model.to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    if last_n_layers > 0:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = list(model.neck.parameters()) + list(model.classification_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.01},
            {"params": head_params, "lr": lr},
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Loss (compute pos ratio from train split only) ───────────────────────
    train_labels = [full_train_dataset.labels[i] for i in train_dataset.indices]
    train_masks = [full_train_dataset.masks[i] for i in train_dataset.indices]
    n_positive = np.array(train_labels).sum()
    n_total = np.array(train_masks).sum()
    pos_ratio = float(n_positive) / float(n_total)
    alpha = torch.tensor(1.0 - pos_ratio, dtype=torch.float32)
    criterion = WeightedFocalLoss(alpha=alpha, gamma=2.0)
    print(f"pos_ratio={pos_ratio:.4f}, alpha={alpha:.4f}")

    # ── Schedulers ───────────────────────────────────────────────────────────
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_epochs])

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb.init(
        project="active-site-prediction-squidly",
        dir=run_dir,
        name=f"{neck_type}_squidly",
        mode="offline",
    )
    wandb.config.update({
        "neck_type": neck_type,
        "last_n_layers": last_n_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "alpha": alpha.item(),
        "gamma": 2.0,
        "total_epochs": total_epochs,
        "warmup_epochs": warmup_epochs,
        "val_split": val_split,
        "train_size": train_size,
        "val_size": val_size,
    })

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_mcc = -1.0
    best_val_results = None
    best_checkpoint = os.path.join(model_dir, "best_model.pt")

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        train_loss, train_results = train_one_epoch(model, criterion, optimizer, train_loader, device, last_n_layers=last_n_layers)
        scheduler.step()

        val_loss, val_results = evaluate(model, val_loader, criterion, device, last_n_layers=last_n_layers, threshold=None)

        print(f"\ntrain_loss={train_loss:.4f} train_mcc={train_results['mcc']:.4f} | "
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
                "val_metrics": val_results,
                "val_threshold": val_results["threshold"],
            }, best_checkpoint)

    # ── Test on Uni3175 using val threshold from best checkpoint ─────────────
    print(f"\nLoading best checkpoint (val MCC={best_val_mcc:.4f}) for Uni3175 test...")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_threshold = checkpoint["val_threshold"]

    test_loss, test_results = evaluate(model, test_loader, criterion, device, last_n_layers=last_n_layers, threshold=val_threshold)

    print(f"\nUni3175 Test — MCC={test_results['mcc']:.4f} | F1={test_results['f1']:.4f} | "
          f"Precision={test_results['precision']:.4f} | Recall={test_results['recall']:.4f} | "
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

    print_final_summary(best_val_results, test_results, output_file=os.path.join(run_dir, f"final_summary_{neck_type}.txt"))

    wandb.finish()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")

    train_squidly(
        device,
        neck_type="identity",
        warmup_epochs=5,
        total_epochs=100,
        head_hidden_dim=512,
        precomputed_root="./data/squidly/precomputed_embeddings",
    )


if __name__ == "__main__":
    main()