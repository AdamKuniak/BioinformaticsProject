from typing import Dict

import pandas as pd
import torch

class ProteinLocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, root="./data/deep_loc_2_0", partition=None, max_length=1024):
        super().__init__()
        dataset = pd.read_csv(root + "/Swissprot_Train_Validation_dataset.csv", sep=",")
        if partition is not None:
            dataset = dataset[dataset["Partition"].isin(partition)].reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length
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