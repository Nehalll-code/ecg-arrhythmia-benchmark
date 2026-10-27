"""Common training script. Only the model differs between runs.

    python train.py --model transformer
    python train.py --model mlp            # needs cache/features.npz
Outputs -> outputs/<model>/{best.pt, val_probs.npy, test_probs.npy, metrics.json}
Test fold is scored once, after training, with thresholds fitted on the val fold.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_data, load_features
from evaluate import best_thresholds, evaluate, print_report
from models import build_model, input_kind


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(device))).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp", "cnn", "bilstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data", default=None, help="override cache path (e.g. for smoke tests)")
    ap.add_argument("--model-kw", default="{}", help="JSON of model kwargs, e.g. {\"d_model\":128}")
    a = ap.parse_args()

    seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind = input_kind(a.model)
    loader_fn = load_data if kind == "signal" else load_features
    d = loader_fn(a.data) if a.data else loader_fn()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = d["train"], d["val"], d["test"]
    kw = json.loads(a.model_kw)
    if kind == "features":  # standardise with train statistics only
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xva, Xte = [((x - mu) / sd).astype(np.float32) for x in (Xtr, Xva, Xte)]
        kw.setdefault("in_dim", Xtr.shape[1])
        feat_norm = (mu, sd)

    model = build_model(a.model, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{a.model}: {n_params:,} params on {device}; train {len(Xtr)} val {len(Xva)} test {len(Xte)}")

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=a.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    out_dir = os.path.join(a.out, a.model)
    os.makedirs(out_dir, exist_ok=True)
    if kind == "features":  # needed to score external cohorts with the identical transform
        np.savez(os.path.join(out_dir, "feat_norm.npz"), mu=feat_norm[0], sd=feat_norm[1])

    best, bad, history = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        va = evaluate(yva, predict(model, Xva, device))["macro_auroc"]
        history.append({"epoch": ep, "train_loss": tot / len(loader), "val_auroc": va})
        print(f"ep {ep:02d} loss {tot / len(loader):.4f} val AUROC {va:.4f} ({time.time() - t0:.0f}s)")
        if va > best:
            best, bad = va, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    pva, pte = predict(model, Xva, device), predict(model, Xte, device)
    np.save(os.path.join(out_dir, "val_probs.npy"), pva)
    np.save(os.path.join(out_dir, "test_probs.npy"), pte)
    th = best_thresholds(yva, pva)
    res = {"model": a.model, "params": n_params, "args": vars(a), "model_kw": kw, "history": history,
           "val": evaluate(yva, pva, th), "test": evaluate(yte, pte, th)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print_report(a.model + " (test)", res["test"])


if __name__ == "__main__":
    main()
