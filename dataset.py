from pathlib import Path

import pandas as pd
import torch
from numpy import dtype
from torch.utils.data import Dataset


class CoraDataset(Dataset):
    n_nodes = 2708
    n_features = 1433
    n_classes = 7

    def __init__(self, datadir, *, device=None):
        datadir = Path(datadir)

        content = pd.read_csv(datadir / "content.csv")
        features = torch.zeros(
            self.n_nodes,
            self.n_features,
            dtype=torch.float,
            device=device,
        )
        for i, j in content.values:
            features[i, j] = 1
        features /= features.sum(dim=1)
        self.features = features

        labels = pd.read_csv(datadir / "labels.csv")["label"]
        self.labels = torch.tensor(
            labels.values,
            dtype=torch.long,
            device=device,
        )

        n_nodes = len(labels)
        citations = pd.read_csv(datadir / "citations.csv")
        adjacency = torch.zeros(
            self.n_nodes,
            self.n_nodes,
            dtype=torch.bool,
            device=device,
        )
        for i, j in citations.values:
            adjacency[i, j] = True
        self.adjacency = adjacency

        train_mask = torch.zeros_like(self.labels, dtype=bool, device=device)
        valid_mask = torch.zeros_like(self.labels, dtype=bool, device=device)
        test_mask = torch.zeros_like(self.labels, dtype=bool, device=device)
        train_mask[0:140] = True
        valid_mask[140 : 140 + 500] = True
        test_mask[-1000:] = True
        self.train_mask = train_mask
        self.valid_mask = valid_mask
        self.test_mask = test_mask

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return (
            self.features,
            self.labels,
            self.adjacency,
            self.train_mask,
            self.valid_mask,
            self.test_mask,
        )
