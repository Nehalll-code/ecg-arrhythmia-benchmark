"""Efficiency axis: parameters, FLOPs and CPU inference latency per ECG.

    python efficiency.py                # -> outputs/efficiency.json
FLOPs = 2 x multiply-accumulates for one record (torch FlopCounterMode covers conv / linear /
attention; LSTM is added analytically as the counter does not see it). Latency = batch 1,
single CPU thread, median of N timed forwards after warm-up. The MLP additionally pays for
hand-crafted feature extraction, so its end-to-end latency includes that cost.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from data import _record_features
from models import build_model, input_kind

MODELS = ["mlp", "cnn", "bilstm", "transformer"]
F_DIM = None  # filled from a real feature vector


def lstm_flops(model, x):
    """Analytic LSTM FLOPs via forward hooks: per step/direction 2*4*h*(in+h) (+bias ignored)."""
    total = [0]

    def hook(mod, inp, _out):
        B, T, _ = inp[0].shape
        d = 2 if mod.bidirectional else 1
        for l in range(mod.num_layers):
            din = mod.input_size if l == 0 else mod.hidden_size * d
            total[0] += B * T * d * 2 * 4 * mod.hidden_size * (din + mod.hidden_size)

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.LSTM)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total[0]


def measure(name, in_dim, n_runs, model_kw=None):
    kw = dict(model_kw or {})
    if input_kind(name) == "features":
        kw.setdefault("in_dim", in_dim)
        x = torch.randn(1, kw["in_dim"])
    else:
        x = torch.randn(1, 1000, 12)
    model = build_model(name, **kw).eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.backends.mha.set_fastpath_enabled(False)  # so attention is visible to the FLOP counter
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    flops = fc.get_total_flops() + lstm_flops(model, x)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return {"params": params, "flops": int(flops), "latency_ms_model": float(np.median(ts)),
            "latency_ms_p95": float(np.percentile(ts, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(1)
    sig = np.random.default_rng(0).standard_normal((1000, 12)).astype(np.float32)
    _record_features(sig)  # warm-up (imports)
    t = []
    for _ in range(30):
        t0 = time.perf_counter()
        feats = _record_features(sig)
        t.append((time.perf_counter() - t0) * 1e3)
    feat_ms, in_dim = float(np.median(t)), len(feats)

    res = {}
    for name in MODELS:
        kw = None
        mpath = os.path.join(a.out, name, "metrics.json")
        if os.path.exists(mpath):  # use the exact tuned config if trained
            with open(mpath) as f:
                kw = json.load(f).get("model_kw")
        r = measure(name, in_dim, a.runs, kw)
        r["latency_ms_end_to_end"] = r["latency_ms_model"] + (feat_ms if name == "mlp" else 0.0)
        res[name] = r
        print(f"{name:12s} params {r['params']:>9,}  MFLOPs {r['flops'] / 1e6:8.1f}  "
              f"model {r['latency_ms_model']:6.2f} ms  end-to-end {r['latency_ms_end_to_end']:6.2f} ms")
    res["_feature_extraction_ms"] = feat_ms
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "efficiency.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
