import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_utils import UniprotDataset, MCSADataset
import os
from tqdm import tqdm
import argparse


def precompute_embeddings(dataset="uniprot", batch_size=16, pretrained_model="facebook/esm2_t33_650M_UR50D", max_length=1024, flush_every=50, last_n_layers: int = 0):
    """
    Precompute ESM-2 embeddings up to the last frozen layer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Dataset: {dataset}, Fine-tunning last {last_n_layers} layers")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    backbone = EsmModel.from_pretrained(pretrained_model).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    total_layers = len(backbone.encoder.layer)  # 33 for 650M
    hidden_dim = backbone.config.hidden_size   # 1280 for 650M

    # Validate fine-tuned layers
    if last_n_layers < 0:
        assert 0 < last_n_layers < total_layers, f"freeze_layers must be between 1 and {total_layers-1}, got {last_n_layers}"
        print(f"  Saving output of fine-tuning last {last_n_layers} layers")
    else:
        print(f"  Saving full backbone output (all {total_layers} layers frozen)")

    # Dataset and output directory
    if dataset == "uniprot":
        output_dir = (f"./data/uniprot/precomputed_embeddings_last_{last_n_layers}_unfrozen" if last_n_layers > 0 else "./data/uniprot/precomputed_embeddings")
        dataset = UniprotDataset(tokenizer, fold=None, max_length=max_length)
    elif dataset == "m-csa":
        output_dir = (f"./data/m_csa/precomputed_embeddings_last_{last_n_layers}_unfrozen" if last_n_layers > 0 else "./data/m_csa/precomputed_embeddings")
        dataset = MCSADataset(tokenizer, max_length=max_length)
    elif dataset == "squidly_14230":
        output_dir = (f"./data/squidly/precomputed_embeddings_last_{last_n_layers}_unfrozen" if last_n_layers > 0 else "./data/squidly/precomputed_embeddings")
        dataset = UniprotDataset(tokenizer, root="./data/squidly/uni14230_clean.json", fold=None, max_length=max_length)
    elif dataset == "squidly_3175":
        output_dir = (f"./data/squidly/precomputed_embeddings_last_{last_n_layers}_unfrozen" if last_n_layers > 0 else "./data/squidly/precomputed_embeddings")
        dataset = UniprotDataset(tokenizer, root="./data/squidly/uni3175_clean.json", fold=None, max_length=max_length)
    else:
        raise ValueError(f"Unknown mode: '{dataset}'. Choose 'uniprot' or 'm-csa'.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    total = len(dataset)

    os.makedirs(output_dir, exist_ok=True)

    def pad_label(label):
        t = torch.tensor(label, dtype=torch.float)
        t = t[:max_length]
        pad_length = max_length - t.size(0)
        if pad_length > 0:
            t = torch.nn.functional.pad(t, (0, pad_length), value=0)
        return t

    # metadata
    metadata = {
        "labels": torch.stack([pad_label(rec["label"]) for rec in dataset.data]),
        "fold": np.array([rec["fold"] for rec in dataset.data]),
        "length": total,
        "max_length": max_length,
        "hidden_dim": hidden_dim,
        "last_n_layers": last_n_layers,
    }

    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    # create memory-mapped files for embeddings and masks
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

            if last_n_layers == 0:
                # full backbone frozen
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state

            else:
                # partially frozen backbone
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # +1 because index 0 is the embedding layer, not a transformer layer
                embeddings = outputs.hidden_states[total_layers - last_n_layers]

            # Strip CLS and EOS
            embeddings = embeddings[:, 1:-1, :]
            attention_mask = attention_mask[:, 1:-1]

            # Pad to max_length if shorter
            seq_len = embeddings.size(1)
            if seq_len < max_length:
                pad = max_length - seq_len
                embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, pad))
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad))

            bs = embeddings.size(0)
            emb_mmap[idx:idx + bs] = embeddings.half().cpu().numpy()
            mask_mmap[idx:idx + bs] = attention_mask.cpu().numpy().astype(np.bool_)
            idx += bs

            if (batch_num + 1) % flush_every == 0:
                emb_mmap.flush()
                mask_mmap.flush()

    emb_mmap.flush()
    mask_mmap.flush()
    print(f"Done! Saved {idx} embeddings to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="uniprot", choices=["uniprot", "m-csa", "squidly_3175", "squidly_14230"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--flush_every", type=int, default=50)
    args = parser.parse_args()

    precompute_embeddings(dataset="squidly_3175")
    precompute_embeddings(dataset="squidly_14230")
    precompute_embeddings(dataset="squidly_3175", last_n_layers=1)
    precompute_embeddings(dataset="squidly_14230", last_n_layers=1)
