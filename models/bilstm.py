import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Bidirectional LSTM. Input (B, T, 12). A strided conv front-end (x4 downsample,
    1000 -> 250 steps) keeps BPTT tractable; the recurrence does the temporal modelling."""

    def __init__(self, in_ch=12, n_classes=5, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(in_ch, 32, 7, 2, 3), nn.ReLU(),
                                   nn.Conv1d(32, 64, 7, 2, 3), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):
        x = self.front(x.transpose(1, 2)).transpose(1, 2)  # (B, 250, 64)
        out, _ = self.lstm(x)
        return self.head(torch.cat([out.mean(1), out.amax(1)], dim=1))
