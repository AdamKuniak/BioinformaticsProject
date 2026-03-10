import torch
import numpy as np
from torch.utils.data import DataLoader
from train import compute_metrics, build_neck
from model import ActiveSitePredictor, ActiveSitePredictorHead
from data_utils import PrecomputedMCSADataset
import os

NUM_FOLDS = 5


def evaluate_model(model, test_loader, device, threshold, last_n_layers=None):
    model.eval()

    all_labels = []
    all_logits = []
    all_masks = []

    with torch.inference_mode():
        for batch in test_loader:
            embeddings = batch['embedding'].to(device, dtype=torch.float32)
            labels = batch['label'].to(device, dtype=torch.float32)
            mask = batch['attention_mask'].to(device, dtype=torch.float32)

            padding_mask = (mask == 0)

            if last_n_layers is not None:
                logits = model(embeddings, mask, last_n_layers=last_n_layers)
            else:
                logits = model(embeddings, mask)

            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            all_masks.append(padding_mask.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    return compute_metrics(all_logits, all_labels, all_masks, threshold)


def load_model(path, neck_type, device, last_n_layers=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    neck_dropout = checkpoint.get("neck_dropout", 0.1)
    neck = build_neck(neck_type, dropout=neck_dropout)

    if last_n_layers is not None:
        model = ActiveSitePredictor(neck=neck, head_hidden_dim=512)
        # model = ActiveSitePredictor(neck=neck, head_hidden_dim=checkpoint["head_hidden_dim"])
    else:
        model = ActiveSitePredictorHead(neck=neck, head_hidden_dim=512)
        # model = ActiveSitePredictorHead(neck=neck, head_hidden_dim=checkpoint["head_hidden_dim"])

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    threshold = checkpoint["threshold"]

    return model, threshold


def save_summary(all_fold_results, neck_type, last_n_layers, output_path="./test_results"):
    metrics_to_report = ["mcc", "f1", "precision", "recall", "auprc"]

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Generate output file name
    output_file = os.path.join(output_path, f"test_results_{neck_type}_{last_n_layers}.txt")

    with open(output_file, "w") as f:
        header = (f"\n{'#' * 20} ACTIVE SITE TEST RESULTS — "
                  f"neck={neck_type} last_n_layers={last_n_layers} "
                  f"{'#' * 20}\n")
        print(header)
        f.write(header + "\n")

        for m in metrics_to_report:
            values = [res[m] for res in all_fold_results]
            mean = np.mean(values)
            std = np.std(values)
            line = f"  {m:12}: {mean:.4f} +/- {std:.4f}"
            print(line)
            f.write(line + "\n")

        thresh_header = "\nOptimal thresholds per fold:"
        print(thresh_header)
        f.write(thresh_header + "\n")

        for fold_idx, res in enumerate(all_fold_results):
            line = f"  Fold {fold_idx}: threshold={res['threshold']:.3f}"
            print(line)
            f.write(line + "\n")

    print(f"\nSummary saved to {output_file}")


def test_all_folds(neck_type, last_n_layers, model_root, embed_root="./data/test/precomputed_embeddings", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_subdir = os.path.join(model_root, "models")
    all_fold_results = []

    test_dataset = PrecomputedMCSADataset(root=embed_root)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    assert len(test_dataset) > 0, "Test dataset is empty"

    for fold in range(NUM_FOLDS):
        print(f"\nEvaluating fold {fold}...")

        checkpoint_path = os.path.join(model_subdir, f"best_model_fold_{fold}.pt")
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

        model, threshold = load_model(checkpoint_path, neck_type, device, last_n_layers)

        print(f"  Loaded checkpoint — threshold={threshold:.3f}")

        results = evaluate_model(model, test_loader, device, threshold, last_n_layers)

        # Store threshold alongside metrics for summary
        results["threshold"] = threshold
        all_fold_results.append(results)

        print(f"  Fold {fold} | MCC={results['mcc']:.4f} "
              f"F1={results['f1']:.4f} "
              f"AUPRC={results['auprc']:.4f} "
              f"Precision={results['precision']:.4f} "
              f"Recall={results['recall']:.4f}")

    save_summary(all_fold_results, neck_type, last_n_layers)


def main():
    test_all_folds(neck_type="identity", last_n_layers=1, model_root="./results/20260305_004648_identity_finetune_last_1", embed_root="./data/test/precomputed_embeddings_last_1_frozen", batch_size=32)


if __name__ == "__main__":
    main()
