import torch
import json
import os
import numpy as np


class MCSADataset(torch.utils.data.Dataset):
    """
    Dataset for the M-CSA test set, which contains sequences and binary labels indicating whether each residue is part of an active site.
    """
    def __init__(self, tokenizer, root="./data/test/test_dataset.json", max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = json.load(open(root, "r"))

        self.sequences = [rec["sequence"] for rec in self.data]
        self.labels = [rec["label"] for rec in self.data]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        tokenized_seq = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Pad or truncate label to max_length
        label = label[:self.max_length]
        pad_length = self.max_length - label.size(0)
        if pad_length > 0:
            label = torch.nn.functional.pad(label, (0, pad_length), value=0)

        return {
            "input_ids": tokenized_seq["input_ids"].squeeze(0),
            "attention_mask": tokenized_seq["attention_mask"].squeeze(0),
            "label": label
        }


class UniprotDataset(torch.utils.data.Dataset):
    """
    Uniprot train/validation dataset, which contains sequences and binary labels indicating whether each residue is part of an active site.
    """
    def __init__(self, tokenizer, root="./data/train_val/train_val_dataset.json", fold=None, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = json.load(open(root, "r"))

        if fold is not None:
            self.data = [rec for rec in self.data if rec["fold"] in fold]

        self.sequences = [rec["sequence"] for rec in self.data]
        self.labels = [rec["label"] for rec in self.data]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        tokenized_seq = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Pad or truncate label to max_length to match the sequence
        label = label[:self.max_length]  # truncate if too long
        pad_length = self.max_length - label.size(0)
        if pad_length > 0:
            label = torch.nn.functional.pad(label, (0, pad_length), value=0)

        return {
            "input_ids": tokenized_seq["input_ids"].squeeze(0),
            "attention_mask": tokenized_seq["attention_mask"].squeeze(0),
            "label": label
        }


class PrecomputedUniprotDataset(torch.utils.data.Dataset):
    """
    Dataset that loads precomputed ESM-2 embeddings from memory-mapped files for the Uniprot dataset.
    Run precompute_embeddings.py first to generate the required files.
    """
    def __init__(self, fold=None, root="./data/train_val/precomputed_embeddings"):
        super().__init__()
        metadata = torch.load(os.path.join(root, "metadata.pt"), weights_only=False)
        self._total = metadata["length"]
        self._max_length = metadata["max_length"]
        self._hidden_dim = metadata["hidden_dim"]
        self._root = root

        all_folds = metadata["fold"]  # list of ints
        all_labels = metadata["labels"]  # tensor of shape (total, max_length)

        if fold is not None:
            mask = np.isin(all_folds, fold)
            self.indices = np.where(mask)[0]
        else:
            self.indices = np.arange(self._total)

        self.labels = all_labels[self.indices]

        self._embeddings = None
        self._masks = None

    def open_memmaps(self):
        self._embeddings = np.memmap(
            os.path.join(self._root, "embeddings.dat"),
            dtype=np.float16, mode="r", shape=(self._total, self._max_length, self._hidden_dim)
        )
        self._masks = np.memmap(
            os.path.join(self._root, "masks.dat"),
            dtype=np.bool_, mode="r", shape=(self._total, self._max_length)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._embeddings is None:
            self.open_memmaps()
        real_idx = self.indices[idx]
        return {
            "embedding": torch.from_numpy(self._embeddings[real_idx].copy()),
            "attention_mask": torch.from_numpy(self._masks[real_idx].copy()).long(),
            "label": self.labels[idx].float(),
        }

    @property
    def masks(self):
        return self._masks
