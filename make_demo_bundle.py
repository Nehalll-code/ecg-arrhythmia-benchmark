"""Package everything the Streamlit demo needs into demo/ (small enough to commit and deploy).

    python make_demo_bundle.py          # after run_all.py has produced outputs/ and results/
demo/sample.npz        random PTB-XL test + Georgia ECGs (seeded) with true labels
demo/models/<m>/       best.pt, metrics.json (+ feat_norm.npz for the MLP)
demo/results/          results.md, summary.json, figures; demo/efficiency.json
"""
import argparse
import glob
import os
import shutil

import numpy as np

from data import load_data
from external import EXT_SIGNAL_CACHE, load_external

MODELS = ["mlp", "cnn", "bilstm", "transformer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo")
    ap.add_argument("--n-ptbxl", type=int, default=150)
    ap.add_argument("--n-georgia", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    Xt, yt = load_data()["test"]
    it = np.sort(rng.choice(len(Xt), min(a.n_ptbxl, len(Xt)), replace=False))
    sample = {"X_ptbxl": Xt[it], "y_ptbxl": yt[it], "idx_ptbxl": it}
    del Xt
    if os.path.exists(EXT_SIGNAL_CACHE):
        Xg, yg = load_external()
        ig = np.sort(rng.choice(len(Xg), min(a.n_georgia, len(Xg)), replace=False))
        sample.update(X_georgia=Xg[ig], y_georgia=yg[ig], idx_georgia=ig)
    np.savez_compressed(os.path.join(a.out, "sample.npz"), **sample)

    for m in MODELS:
        src = os.path.join("outputs", m)
        if not os.path.exists(os.path.join(src, "best.pt")):
            print(f"[skip] {m}: no checkpoint")
            continue
        dst = os.path.join(a.out, "models", m)
        os.makedirs(dst, exist_ok=True)
        for f in ("best.pt", "metrics.json", "feat_norm.npz"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), dst)
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    for f in glob.glob("results/*.png") + glob.glob("results/results.md") + glob.glob("results/summary.json"):
        shutil.copy(f, os.path.join(a.out, "results"))
    if os.path.exists("outputs/efficiency.json"):
        shutil.copy("outputs/efficiency.json", a.out)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.out, "**", "*"), recursive=True)
               if os.path.isfile(p))
    print(f"demo bundle -> {a.out}/ ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
