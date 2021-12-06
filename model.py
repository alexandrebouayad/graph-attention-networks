import numpy as np
import torch
import torch.nn as nn

DROPOUT_PROBA = 0.6
LEAKY_RELU_SLOPE = 0.2


class GraphAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        conv1d = nn.Conv1d(in_features, 2, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(conv1d.weight)
        self.linear_layer = conv1d
        self.leaky_relu = nn.LeakyReLU(LEAKY_RELU_SLOPE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h, adjacency):
        e = self.linear_layer(h)
        e1, e2 = e.split(1, dim=1)
        e = e1.transpose(1, 2) + e2
        e = self.leaky_relu(e)
        adjacency |= torch.eye(adjacency.shape[0], dtype=torch.bool)
        e[~adjacency] = float("-inf")
        return self.softmax(e)


class GraphAttentionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        conv1d = nn.Conv1d(in_features, out_features, 1, bias=False)
        nn.init.xavier_uniform_(conv1d.weight)
        self.linear_layer = conv1d
        self.attention = GraphAttention(out_features)
        self.dropout = nn.Dropout(DROPOUT_PROBA)

    def forward(self, h, adjacency):
        h = self.dropout(h)
        h = self.linear_layer(h)
        e = self.attention(h, adjacency)
        e = self.dropout(e)
        return h @ e


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, *, final=False):
        super().__init__()
        self.heads = nn.ModuleList(
            GraphAttentionHead(in_features, out_features)
            for _ in range(n_heads)
        )
        self.elu = nn.ELU()
        self.final = final

    def forward(self, h, adjacency):
        h = torch.stack([head(h, adjacency) for head in self.heads])
        if self.final:
            return self.elu(h.mean(dim=0))
        h = self.elu(h)
        return torch.cat(list(h), dim=1)


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, hidden_units, n_classes, n_heads):
        super().__init__()
        self.first_layer = GraphAttentionLayer(
            in_features, hidden_units, n_heads[0]
        )
        self.second_layer = GraphAttentionLayer(
            hidden_units * n_heads[0], n_classes, n_heads[1], final=True
        )

    def forward(self, h, adjacency):
        h = h.transpose(1, 2)
        h = self.first_layer(h, adjacency)
        h = self.second_layer(h, adjacency)
        return h.transpose(1, 2)
