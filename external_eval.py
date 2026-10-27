"""Zero-shot external validation: run the trained PTB-XL models, UNCHANGED, on the Georgia cohort.

    python external_eval.py                      # all models found in outputs/
Thresholds are the PTB-XL validation-fold thresholds stored in metrics.json (nothing is re-tuned).
Writes outputs/<model>/{external.json, ext_probs.npy}.
"""
import argparse
import json
import os

import numpy as np
import torch

from evaluate import evaluate, print_report
from external import EXT_FEATURE_CACHE, EXT_SIGNAL_CACHE, load_external
from models import build_model, input_kind
from train import predict

MODELS = ["mlp", "cnn", "bilstm", "transformer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--signal-cache", default=EXT_SIGNAL_CACHE)
    ap.add_argument("--feature-cache", default=EXT_FEATURE_CACHE)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in MODELS:
        d = os.path.join(a.out, name)
        if not os.path.exists(os.path.join(d, "best.pt")):
            print(f"[skip] {name}: no checkpoint in {d}")
            continue
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        if input_kind(name) == "signal":
            X, y = load_external(a.signal_cache)
        else:
            X, y = load_external(a.feature_cache)
            n = np.load(os.path.join(d, "feat_norm.npz"))
            X = ((X - n["mu"]) / n["sd"]).astype(np.float32)
        model = build_model(name, **m["model_kw"]).to(device)
        model.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device))
        probs = predict(model, X, device)
        np.save(os.path.join(d, "ext_probs.npy"), probs)
        res = evaluate(y, probs, np.array(m["val"]["thresholds"]))
        res["n_records"] = int(len(y))
        with open(os.path.join(d, "external.json"), "w") as f:
            json.dump(res, f, indent=2)
        print_report(f"{name} (Georgia, external)", res)


if __name__ == "__main__":
    main()
