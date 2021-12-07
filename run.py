import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CoraDataset
from model import GraphAttentionNetwork

DATADIR = "data/cora"

HIDDEN_UNITS = 8
N_HEADS = [8, 1]
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
N_EPOCHS = 100_000
PATIENCE = 100


def accuracy(logits, labels):
    accuracy = logits.argmax(dim=1) == labels
    return accuracy.float().mean().item()


def train(dataloader, *, model, loss_fn, optimizer, patience):
    min_valid_loss = float("inf")
    max_valid_accuracy = 0
    stop_counter = 0

    for features, labels, adjacency, train_mask, valid_mask, _ in dataloader:
        if stop_counter == patience:
            break

        model.train()
        pred = model(features, adjacency)
        train_pred = pred[train_mask]
        train_labels = labels[train_mask]
        train_loss = loss_fn(train_pred, train_labels)
        train_accuracy = accuracy(train_pred, train_labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_pred = pred[valid_mask]
            valid_labels = labels[valid_mask]
            valid_loss = loss_fn(valid_pred, valid_labels)
            valid_accuracy = accuracy(valid_pred, valid_labels)

        if valid_loss > min_valid_loss and valid_accuracy < max_valid_accuracy:
            stop_counter += 1
        else:
            stop_counter = 0
            min_valid_loss = min(min_valid_loss, valid_loss)
            max_valid_accuracy = max(max_valid_accuracy, valid_accuracy)

    return train_loss, train_accuracy, valid_loss, valid_accuracy


@torch.no_grad()
def test(dataloader, *, model, loss_fn):
    model.eval()
    for features, labels, adjacency, _, _, test_mask in dataloader:
        pred = model(features, adjacency)
        test_pred = pred[test_mask]
        test_labels = labels[test_mask]
        test_loss = loss_fn(test_pred, test_labels)
        test_accuracy = accuracy(test_pred, test_labels)
    return test_loss, test_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CoraDataset(DATADIR, device=device)
    dataloader = DataLoader(dataset)

    model = GraphAttentionNetwork(
        in_features=dataset.n_features,
        hidden_units=HIDDEN_UNITS,
        n_classes=dataset.n_classes,
        n_heads=N_HEADS,
    )
    model.to(device)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(N_EPOCHS):
        train_loss, train_accuracy, valid_loss, valid_accuracy = train(
            dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            patience=PATIENCE,
        )
        test_loss, test_accuracy = test(
            dataloader,
            model=model,
            loss_fn=loss_fn,
        )
        print(
            f"Epoch: {epoch + 1:05d}",
            f"train: ({train_loss:.3f}, {train_accuracy:.3f})",
            f"validation: ({valid_loss:.3f}, {valid_accuracy:.3f})",
            f"test: ({test_loss:.3f}, {test_accuracy:.3f})",
            sep=", ",
        )


if __name__ == "main":
    main()
