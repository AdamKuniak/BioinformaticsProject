import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
import pandas as pd
from data_utils import SwissProtDataset
from torch.utils.data import DataLoader
import os

def precompute(batch_size=8, output_dir="./data/precomputed_embeddings", pretrained_model="facebook/esm2_t33_650M_UR50D", max_length=1024):
    """
    Precompute ESM-2 embeddings for the SwissProt dataset and save them to memory-mapped files for efficient loading during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the backbone
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    backbone = EsmModel.from_pretrained(pretrained_model).to(device)
    backbone.eval()

    hidden_dim = backbone.config.hidden_size  # 1280 for esm2_t33_650M_UR50D

    # Load SwissProt dataset
    dataset = SwissProtDataset(tokenizer, partition=None, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    total = len(dataset)

    os.makedirs(output_dir, exist_ok=True)

    # Save metadata (partitions + labels) for PrecomputedDataset to filter by fold
    df = pd.read_csv("./data/Swissprot_Train_Validation_dataset.csv", sep=",")
    label_columns = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
                     "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]
    torch.save({
            "partitions": df["Partition"].values,
            "labels": df[label_columns].values,
            "length": total,
            "max_length": max_length,
            "hidden_dim": hidden_dim
        }, os.path.join(output_dir, "metadata.pt")
    )

    # Memmap files for embeddings and attention masks
    emb_mmap = np.memmap(
        os.path.join(output_dir, "embeddings.dat"),
        dtype=np.float16, mode="w+", shape=(total, max_length, hidden_dim)
    )
    mask_mmap = np.memmap(
        os.path.join(output_dir, "masks.dat"),
        dtype=np.bool_, mode="w+", shape=(total, max_length)
    )

    print(f"Precomputing embeddings for {total} samples...")
    idx = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch_num, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

            bs = embeddings.size(0)
            # Save to memmap (convert to half precision and move to CPU first for faster saving)
            emb_mmap[idx:idx + bs] = embeddings.half().cpu().numpy()
            mask_mmap[idx:idx + bs] = attention_mask.cpu().numpy().astype(np.int8)
            idx += bs

            if (batch_num + 1) % 10 == 0:
                print(f"  {idx}/{total} ({100 * idx / total:.1f}%)")

    # Flush memmaps to ensure data is written to disk
    emb_mmap.flush()
    mask_mmap.flush()
    print(f"Done! Saved {idx} embeddings to {output_dir}")

if __name__ == "__main__":
    precompute()




