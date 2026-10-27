import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, cin, cout, stride, k=7, dropout=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        out = self.drop(torch.relu(self.b1(self.c1(x))))
        out = self.b2(self.c2(out))
        return torch.relu(out + (x if self.skip is None else self.skip(x)))


class ResNet1D(nn.Module):
    """Residual 1D-CNN. Input (B, T, 12) -> logits (B, 5)."""

    def __init__(self, in_ch=12, n_classes=5, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, widths[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU())
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(_Block(c, w, stride=1 if i == 0 else 2))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(c * 2, n_classes)  # concat avg+max pooling

    def forward(self, x):
        x = self.blocks(self.stem(x.transpose(1, 2)))
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], dim=1))
