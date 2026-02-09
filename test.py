import torch
from data_utils import HPADataset
from model import ProteinLocalizator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torchmetrics
import wandb
from focal_loss import MultiLabelFocalLoss
import json

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="DeepLoc2-Replication", job_type="final-test", name="HPA-Benchmark")

    # Model Setup
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = ProteinLocalizator().to(device)

    # Load Weights
    checkpoint = torch.load("best_model.pt", map_location=device)
    model.attention_pooling.load_state_dict(checkpoint["attention_pooling"])
    model.classifier_head.load_state_dict(checkpoint["classifier"])

    # Move thresholds to device for comparison
    best_thresholds = checkpoint["thresholds"].to(device)
    model.eval()

    # Data Setup
    test_dataset = HPADataset(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = MultiLabelFocalLoss().to(device)

    # Metrics
    all_preds = []
    all_labels = []
    total_loss = 0.0

    print("Running inference on HPA dataset")
    with torch.inference_mode():
        for batch in test_loader:
            data = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(data, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits)

            # Accumulate EVERYTHING for the final calculation
            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = total_loss / len(test_loader)

    # Concatenate all batches into two giant tensors
    all_preds = torch.cat(all_preds, dim=0)  # Shape: [1717, 10]
    all_labels = torch.cat(all_labels, dim=0)  # Shape: [1717, 10]

    # Apply the saved thresholds
    binary_preds = (all_preds >= best_thresholds).long()

    # Calculate Metrics
    # MCC per class
    mcc_per_class = torchmetrics.functional.matthews_corrcoef(binary_preds, all_labels.long(), task="multilabel", num_labels=10)
    mcc_macro = mcc_per_class.mean().item()
    acc = torchmetrics.functional.classification.multilabel_exact_match(binary_preds, all_labels.long(), num_labels=10).item()
    f1_score = torchmetrics.functional.f1_score(binary_preds, all_labels.long(), num_labels=10, task='multilabel').item()
    jaccard = torchmetrics.functional.jaccard_index(binary_preds, all_labels.long(), num_labels=10, task='multilabel').item()

    # Save and Print Results
    results = {
        "test_loss": avg_loss,
        "test_mcc_macro": mcc_macro,
        "test_acc": acc,
        "test_mcc_per_class": mcc_per_class.tolist(),
        "f1_score": f1_score,
        "jaccard": jaccard
    }

    # Print to terminal
    print(f"\n{'=' * 30}\nFINAL HPA RESULTS\n{'=' * 30}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Macro MCC: {mcc_macro:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1_score: {f1_score:.4f}")
    print(f"Jaccard: {jaccard:.4f}")

    # Save to JSON for your records/paper
    with open("hpa_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Log to wandb
    wandb.log(results)
    wandb.finish()


if __name__ == '__main__':
    main()