import torch
import json


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

        return {
            "input_ids": tokenized_seq["input_ids"].squeeze(0),
            "attention_mask": tokenized_seq["attention_mask"].squeeze(0),
            "label": label
        }