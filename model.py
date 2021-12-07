import torch
import torch.nn as nn

DROPOUT_PROBA = 0.6
LEAKY_RELU_SLOPE = 0.2


class GraphAttention(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        a = torch.empty(n_features, 2)
        self.a = nn.Parameter(a)
        self.leaky_relu = nn.LeakyReLU(LEAKY_RELU_SLOPE)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, h, adjacency):
        e = h @ self.a
        e1, e2 = e.split(1, dim=1)
        e = e1 + e2.T
        e = self.leaky_relu(e)
        e[~adjacency] = float("-inf")
        return self.softmax(e)


class GraphAttentionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = torch.empty(in_features, out_features)
        self.w = nn.Parameter(w)
        self.attention = GraphAttention(out_features)
        self.dropout = nn.Dropout(DROPOUT_PROBA)
        nn.init.xavier_uniform_(self.w.data)

    def forward(self, h, adjacency):
        h = h @ self.w
        e = self.attention(h, adjacency)
        e = self.dropout(e)
        return e @ h


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, *, final=False):
        super().__init__()
        self.heads = nn.ModuleList(
            GraphAttentionHead(in_features, out_features)
            for _ in range(n_heads)
        )
        self.final = final

    def forward(self, h, adjacency):
        h_list = [head(h, adjacency) for head in self.heads]
        if self.final:
            return torch.stack(h_list).mean(dim=0)
        return torch.cat(h_list, dim=1)


class GraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_units,
        n_classes,
        n_heads,
        bias=False,
    ):
        super().__init__()
        self.first_layer = GraphAttentionLayer(
            in_features,
            hidden_units,
            n_heads[0],
        )
        self.second_layer = GraphAttentionLayer(
            hidden_units * n_heads[0],
            n_classes,
            n_heads[1],
            final=True,
        )
        self.dropout = nn.Dropout(DROPOUT_PROBA)
        self.elu = nn.ELU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        if bias:
            first_bias = torch.zeros(hidden_units * n_heads[0])
            second_bias = torch.zeros(n_classes)
            self.first_bias = nn.Parameter(first_bias)
            self.second_bias = nn.Parameter(second_bias)
        self.bias = bias

    def forward(self, h, adjacency):
        adjacency_diag = adjacency.diagonal()
        adjacency_diag[:] = True

        h = self.dropout(h)
        h = self.first_layer(h, adjacency)
        if self.bias:
            h += self.first_bias
        h = self.elu(h)
        h = self.dropout(h)
        h = self.second_layer(h, adjacency)
        if self.bias:
            h += self.second_bias
        return self.log_softmax(h)
