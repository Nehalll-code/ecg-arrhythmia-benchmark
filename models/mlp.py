import torch.nn as nn


class MLP(nn.Module):
    """Baseline: feed-forward net on engineered features. Input (B, F)."""

    def __init__(self, in_dim, n_classes=5, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
