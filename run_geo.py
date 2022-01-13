from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            8,
            heads=8,
            dropout=0.6,
            bias=False,
        )
        self.conv2 = GATConv(
            8 ** 2,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.6,
            bias=False,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


with TemporaryDirectory() as tmpdirname:
    dataset = Planetoid("tmp", "cora", transform=T.NormalizeFeatures())
    # dataset = Planetoid("tmp", "cora")
    data = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accuracies = model(data.x, data.edge_index), []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        accuracy = out[mask].argmax(-1) == data.y[mask]
        accuracy = accuracy.float().mean().item()
        accuracies.append(accuracy)
    return accuracies


def main():
    for epoch in range(1, 100_001):
        train(data)
        train_accuracy, valid_accuracy, test_accuracy = test(data)
        print(
            f"Epoch: {epoch:05d}",
            f"train: {train_accuracy:.3f}",
            f"validate: {valid_accuracy:.4f}",
            f"test: {test_accuracy:.3f}",
            sep=", ",
        )


if __name__ == "__main__":
    main()
