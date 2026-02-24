import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_utils import UniprotDataset, MCSADataset
import os
from tqdm import tqdm
import argparse


def precompute_embeddings(mode="train", batch_size=16, pretrained_model="facebook/esm2_t33_650M_UR50D", max_length=1024, flush_every=50):
    """
    Precompute ESM-2 embeddings and save them to memory-mapped files.
    mode: "train" for SwissProt train/val dataset, "test" for HPA test dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Mode: {mode}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    backbone = EsmModel.from_pretrained(pretrained_model).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    hidden_dim = backbone.config.hidden_size

    if mode == "train":
        input_file = "./data/train_val/train_val_dataset.json"
        output_dir = "./data/train_val/precomputed_embeddings"
        dataset = UniprotDataset(tokenizer, fold=None, max_length=max_length)
    elif mode == "test":
        input_file = "./data/test/test_dataset.json"
        output_dir = "./data/test/precomputed_hpa_embeddings"
        dataset = MCSADataset(tokenizer, max_length=max_length)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'train' or 'test'.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    total = len(dataset)

    os.makedirs(output_dir, exist_ok=True)

    # Build and save metadata
    if mode == "train":
        df = pd.read_json(input_file)
        metadata = {
            "fold": torch.tensor(df["fold"].values),
            "length": total,
            "max_length": max_length,
            "hidden_dim": hidden_dim
        }
    else:
        metadata = {
            "total": total,
            "max_length": max_length,
            "hidden_dim": hidden_dim,
            "labels": torch.tensor(dataset.labels, dtype=torch.float)
        }

    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    emb_mmap = np.memmap(
        os.path.join(output_dir, "embeddings.dat"),
        dtype=np.float16, mode="w+", shape=(total, max_length, hidden_dim)
    )
    mask_mmap = np.memmap(
        os.path.join(output_dir, "masks.dat"),
        dtype=np.bool_, mode="w+", shape=(total, max_length)
    )

    print(f"Precomputing embeddings for {total} samples...")
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = 0

    with torch.no_grad(), torch.amp.autocast(autocast_device):
        for batch_num, batch in tqdm(enumerate(loader), total=len(loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)

            embeddings = outputs.last_hidden_state[:, 1:-1, :]
            attention_mask = attention_mask[:, 1:-1]

            embeddings = embeddings[:, :max_length, :]
            attention_mask = attention_mask[:, :max_length]

            bs = embeddings.size(0)
            emb_mmap[idx:idx + bs] = embeddings.half().cpu().numpy()
            mask_mmap[idx:idx + bs] = attention_mask.cpu().numpy().astype(np.int8)
            idx += bs

            if (batch_num + 1) % flush_every == 0:
                print(f"  {idx}/{total} ({100 * idx / total:.1f}%)")
                emb_mmap.flush()
                mask_mmap.flush()

    emb_mmap.flush()
    mask_mmap.flush()
    print(f"Done! Saved {idx} embeddings to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="'train' for SwissProt train/val, 'test' for HPA test set")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--flush_every", type=int, default=50)
    args = parser.parse_args()

    precompute_embeddings(
        mode=args.mode,
        batch_size=args.batch_size,
        max_length=args.max_length,
        flush_every=args.flush_every
    )
