import torch
import torch.nn as nn
import torch.nn.functional as F

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


def train(dataset, *, model, loss_fn, optimizer):
    features, labels, adjacency, train_mask, valid_mask, _ = dataset[0]

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

    return train_loss, train_accuracy, valid_loss, valid_accuracy


@torch.no_grad()
def test(dataset, *, model, loss_fn):
    features, labels, adjacency, _, _, test_mask = dataset[0]

    model.eval()
    pred = model(features, adjacency)
    test_pred = pred[test_mask]
    test_labels = labels[test_mask]
    test_loss = loss_fn(test_pred, test_labels)
    test_accuracy = accuracy(test_pred, test_labels)

    return test_loss, test_accuracy


def main(bias=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CoraDataset(DATADIR, device=device)
    model = GraphAttentionNetwork(
        in_features=dataset.n_features,
        hidden_units=HIDDEN_UNITS,
        n_classes=dataset.n_classes,
        n_heads=N_HEADS,
        bias=bias,
    )
    model.to(device)

    loss_fn = F.nll_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    min_valid_loss = float("inf")
    max_valid_accuracy = 0
    stop_counter = 0

    for epoch in range(N_EPOCHS):
        train_loss, train_accuracy, valid_loss, valid_accuracy = train(
            dataset,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        test_loss, test_accuracy = test(
            dataset,
            model=model,
            loss_fn=loss_fn,
        )

        if valid_loss > min_valid_loss and valid_accuracy < max_valid_accuracy:
            stop_counter += 1
        else:
            stop_counter = 0
            min_valid_loss = min(min_valid_loss, valid_loss)
            max_valid_accuracy = max(max_valid_accuracy, valid_accuracy)

        print(
            f"Epoch: {epoch + 1:05d}",
            f"train: ({train_loss:.3f}, {train_accuracy:.3f})",
            f"validation: ({valid_loss:.3f}, {valid_accuracy:.3f})",
            f"test: ({test_loss:.3f}, {test_accuracy:.3f})",
            f"stop_counter={stop_counter}",
            sep=", ",
        )
        if stop_counter == PATIENCE:
            break


if __name__ == "main":
    main()
