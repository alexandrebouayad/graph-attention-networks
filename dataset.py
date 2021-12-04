from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class CoraDataset(Dataset):
    def __init__(self, datadir, train=True):
        datadir = Path(datadir)
        self.train = train

        words = pd.read_csv(datadir / "words.csv")
        features = pd.get_dummies(words, columns=["word"])
        features = features.groupby("paper").sum()
        self.features = torch.tensor(features.values, dtype=torch.float)

        labels = pd.read_csv(datadir / "labels.csv")["label"]
        self.labels = torch.tensor(labels.values, dtype=torch.long)

        n_nodes = len(labels)
        citations = pd.read_csv(datadir / "citations.csv")
        adjacency = torch.eye(n_nodes, dtype=torch.bool)
        for i, j in citations.values:
            adjacency[i, j] = True
        self.adjacency = adjacency

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        if self.train:
            train_mask = torch.zeros_like(self.labels, dtype=bool)
            valid_mask = torch.zeros_like(self.labels, dtype=bool)
            train_mask[0:140] = True
            valid_mask[140 : 140 + 500] = True
            return (
                self.features,
                self.labels,
                self.adjacency,
                train_mask,
                valid_mask,
            )
        test_mask = torch.zeros_like(self.labels, dtype=bool)
        test_mask[-1000:] = True
        return self.features, self.labels, self.adjacency, test_mask
