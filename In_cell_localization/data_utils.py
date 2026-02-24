import pandas as pd
import torch
import os
import numpy as np

class SwissProtDataset(torch.utils.data.Dataset):
    """
    Class representing a Train/Validation SwissProt dataset
    tokenizer: Tokenizer object to tokenize protein sequences
    partition: list of integers representing the partition for cross validation
    """
    def __init__(self, tokenizer, partition=None, root="./data/", max_length=1024):
        super().__init__()
        dataset = pd.read_csv(root + "/Swissprot_Train_Validation_dataset.csv", sep=",")

        # Take only certain partition of the data
        dataset = dataset[dataset["Partition"].isin(partition)].reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length
        # All the possible protein locations
        self.LABEL_COLUMNS = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

        self.labels = dataset[self.LABEL_COLUMNS].values
        self.sequences = dataset["Sequence"].values

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[index]

        tokenized_seq = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length)

        label = torch.tensor(self.labels[index], dtype=torch.float)

        return {
            "input_ids": tokenized_seq["input_ids"].squeeze(0),
            "attention_mask": tokenized_seq["attention_mask"].squeeze(0),
            "label": label
        }


class PrecomputedSwissDataset(torch.utils.data.Dataset):
    """
        Dataset that loads precomputed ESM-2 embeddings from memory-mapped files.
        Run precompute.py first to generate the required files.
    """
    def __init__(self, partition=None, root="./data/precomputed_embeddings"):
        super().__init__()
        metadata = torch.load(os.path.join(root, "metadata.pt"), weights_only=False)
        self._total = metadata["total"]
        self._max_length = metadata["max_length"]
        self._hidden_dim = metadata["hidden_dim"]
        self._root = root

        all_partitions = metadata["partitions"].numpy()
        all_labels = metadata["labels"].numpy()

        if partition is not None:
            mask = np.isin(all_partitions, partition)
            self.indices = np.where(mask)[0]
        else:
            self.indices = np.arange(self._total)

        self.labels = all_labels[self.indices]

        self._embeddings = None
        self.masks = None

    def open_memmaps(self):
        self._embeddings = np.memmap(
            os.path.join(self._root, "embeddings.dat"),
            dtype=np.float16, mode="r", shape=(self._total, self._max_length, self._hidden_dim)
        )
        self.masks = np.memmap(
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
            "attention_mask": torch.from_numpy(self.masks[real_idx].copy()).long(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class HPADataset(torch.utils.data.Dataset):
    """
        Class representing a test HPA dataset
        tokenizer: Tokenizer object to tokenize protein sequences
    """
    def __init__(self, tokenizer, root="./data/", max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        dataset = pd.read_csv(root + "/hpa_testset.csv", sep=",")
        self.LABEL_COLUMNS = ["Cell membrane", "Cytoplasm", "Endoplasmic reticulum", "Golgi apparatus", "Lysosome/Vacuole", "Mitochondrion", "Nucleus", "Peroxisome"]
        self.labels = dataset[self.LABEL_COLUMNS].values
        self.sequences = dataset["fasta"].values

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[index]
        tokenized_seq = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        label = torch.tensor(self.labels[index], dtype=torch.float)

        return {
            "input_ids": tokenized_seq["input_ids"].squeeze(0),
            "attention_mask": tokenized_seq["attention_mask"].squeeze(0),
            "label": label
        }


class PrecomputedHPADataset(torch.utils.data.Dataset):
    """
        Dataset that loads precomputed ESM-2 embeddings from memory-mapped files for the HPA test set.
        Run precompute.py first to generate the required files.
    """
    def __init__(self, root="./data/test/hpa_precomputed_embeddings"):
        super().__init__()
        metadata = torch.load(os.path.join(root, "metadata.pt"), weights_only=False)
        self._total = metadata["length"]
        self._max_length = metadata["max_length"]
        self._hidden_dim = metadata["hidden_dim"]
        self._root = root

        all_labels = metadata["labels"].numpy()
        self.indices = np.arange(self._total)
        self.labels = all_labels

        self.LABEL_COLUMNS = ["Cell membrane", "Cytoplasm", "Endoplasmic reticulum", "Golgi apparatus", "Lysosome/Vacuole", "Mitochondrion", "Nucleus", "Peroxisome"]

        self._embeddings = None
        self.masks = None

    def open_memmaps(self):
        self._embeddings = np.memmap(
            os.path.join(self._root, "embeddings.dat"),
            dtype=np.float16, mode="r", shape=(self._total, self._max_length, self._hidden_dim)
        )
        self.masks = np.memmap(
            os.path.join(self._root, "masks.dat"),
            dtype=np.bool_, mode="r", shape=(self._total, self._max_length)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._embeddings is None:
            self.open_memmaps()
        return {
            "embedding": torch.from_numpy(self._embeddings[idx].copy()),
            "attention_mask": torch.from_numpy(self.masks[idx].copy()).long(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }
