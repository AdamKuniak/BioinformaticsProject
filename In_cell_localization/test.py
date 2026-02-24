import torch
from data_utils import PrecomputedHPADataset
from model import ProteinLocalizatorHead
from torch.utils.data import DataLoader
import torchmetrics
import numpy as np

COMPARTMENTS = ["Cell membrane", "Cytoplasm", "Endoplasmic reticulum", "Golgi apparatus",
                "Lysosome/Vacuole", "Mitochondrion", "Nucleus", "Peroxisome"]

MODEL_DIR = "results/run_1024_120epo_dp_02/models"
NUM_FOLDS = 5


def evaluate_fold(model, dataloader, device, best_thresholds, hpa_indices):
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            embeddings = batch["embedding"].to(device, dtype=torch.float32)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(embeddings, mask)
            preds = torch.sigmoid(logits)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_preds = all_preds[:, hpa_indices]
    all_labels = torch.cat(all_labels, dim=0)

    binary_preds = (all_preds >= best_thresholds).long()

    mcc_per_class = [
        torchmetrics.functional.matthews_corrcoef(
            binary_preds[:, c], all_labels[:, c].long(), task="binary"
        ).item()
        for c in range(8)
    ]
    mcc_macro = np.mean(mcc_per_class)
    acc = torchmetrics.functional.classification.multilabel_exact_match(
        binary_preds, all_labels.long(), num_labels=8
    ).item()
    f1 = torchmetrics.functional.f1_score(
        binary_preds, all_labels.long(), num_labels=8, task="multilabel"
    ).item()
    jaccard = torchmetrics.functional.jaccard_index(
        binary_preds, all_labels.long(), num_labels=8, task="multilabel"
    ).item()

    return {
        "mcc_macro": mcc_macro,
        "mcc_per_class": mcc_per_class,
        "exact_match_acc": acc,
        "f1_macro": f1,
        "jaccard": jaccard,
    }


def save_summary(all_fold_results, output_file="test_summary.txt"):
    metrics_to_report = ["mcc_macro", "exact_match_acc", "f1_macro", "jaccard"]

    with open(output_file, "w") as f:
        header = f"\n{'#' * 20} HPA TEST RESULTS ACROSS 5 FOLDS {'#' * 20}\n"
        print(header)
        f.write(header + "\n")

        for m in metrics_to_report:
            values = [res[m] for res in all_fold_results]
            mean = np.mean(values)
            std = np.std(values)
            line = f"{m}: {mean:.4f} +/- {std:.4f}"
            print(line)
            f.write(line + "\n")

        per_class_header = "\nPer-Class MCC (mean +/- std):"
        print(per_class_header)
        f.write("\n" + per_class_header + "\n")

        for i, name in enumerate(COMPARTMENTS):
            class_scores = [res["mcc_per_class"][i] for res in all_fold_results]
            line = f"  {name:25}: {np.mean(class_scores):.4f} +/- {np.std(class_scores):.4f}"
            print(line)
            f.write(line + "\n")

    print(f"\nSummary saved to {output_file}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SWISS_COLUMNS = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

    HPA_COLUMNS = ["Cell membrane", "Cytoplasm", "Endoplasmic reticulum", "Golgi apparatus", "Lysosome/Vacuole", "Mitochondrion", "Nucleus", "Peroxisome"]

    # Get the indices in SwissProt order that correspond to each HPA label
    hpa_indices = [SWISS_COLUMNS.index(col) for col in HPA_COLUMNS]

    test_dataset = PrecomputedHPADataset()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    all_fold_results = []

    for fold in range(NUM_FOLDS):
        print(f"\nEvaluating fold {fold}...")
        model = ProteinLocalizatorHead().to(device)

        checkpoint = torch.load(f"{MODEL_DIR}/best_model_fold_{fold}.pt", map_location=device, weights_only=False)
        model.attention_pooling.load_state_dict(checkpoint["attention_pooling"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        best_thresholds = checkpoint["thresholds"].to(device)
        best_thresholds = best_thresholds[hpa_indices]
        model.eval()

        results = evaluate_fold(model, test_loader, device, best_thresholds, hpa_indices)
        all_fold_results.append(results)

        print(f"  Fold {fold} MCC: {results['mcc_macro']:.4f}")

    save_summary(all_fold_results)


if __name__ == "__main__":
    main()