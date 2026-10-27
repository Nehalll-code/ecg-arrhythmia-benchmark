"""One command for the whole experiment, resumable (a model with outputs/<m>/metrics.json is skipped).

    python run_all.py                       # train mlp,cnn,bilstm,transformer -> external eval -> efficiency -> compare
    python run_all.py --models cnn mlp      # subset
    python run_all.py --epochs 30           # extra args are forwarded to train.py
Needs cache/ptbxl_100hz.npz and cache/features.npz (python data.py ... --features); the external
step runs only if data/georgia exists (python download.py georgia; python external.py).
"""
import argparse
import os
import subprocess
import sys

ORDER = ["mlp", "cnn", "bilstm", "transformer"]


def run(*args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    a, train_args = ap.parse_known_args()

    for m in a.models:
        if not a.force and os.path.exists(f"outputs/{m}/metrics.json"):
            print(f"[skip] {m}: outputs/{m}/metrics.json exists (use --force to retrain)")
            continue
        run("train.py", "--model", m, *train_args)

    if os.path.isdir("data/georgia") and not os.path.exists("cache/georgia_100hz.npz"):
        run("external.py", "--data-dir", "data/georgia")
    if os.path.exists("cache/georgia_100hz.npz"):
        run("external_eval.py")
    else:
        print("[skip] external validation: no cache/georgia_100hz.npz")
    run("efficiency.py")
    run("compare.py")
    print("\nDone -> results/results.md (+ figures), outputs/<model>/")


if __name__ == "__main__":
    main()
