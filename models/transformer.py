import torch
import torch.nn as nn


class ECGTransformer(nn.Module):
    """Transformer encoder for 12-lead ECG. Input (B, 1000, 12) -> logits (B, 5).

    Raw samples are not used as tokens (1000-token self-attention is costly and
    each sample carries little information). A strided Conv1d 'patch embedding'
    turns every `patch` samples (default 50 ms at 100 Hz) into one token (200 tokens),
    a learned positional embedding is added, and a CLS token is prepended whose
    final state feeds the classification head. Self-attention then relates
    distant beats/segments directly.
    """

    def __init__(self, in_ch=12, n_classes=5, seq_len=1000, patch=5, d_model=64,
                 n_heads=4, n_layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.embed = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        n_tokens = seq_len // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        t = self.embed(x.transpose(1, 2)).transpose(1, 2)           # (B, N, d)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        return self.head(self.norm(self.encoder(t)[:, 0]))
