"""Scripted, resumable dataset download. Nothing is committed (data/ is git-ignored).

    python download.py ptbxl      # -> data/ptb-xl-1.0.3 (metadata + 100 Hz records only, ~0.3 GB)
    python download.py georgia --limit 3000   # Challenge 2020 Georgia (10,344 recs, 1.2 GB; --limit = seeded subset)

Both are open access (CC BY 4.0 / ODC-By style, no login).
"""
import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PTBXL_BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
PTBXL_DIR = "data/ptb-xl-1.0.3"
GEORGIA_BASE = "https://physionet.org/files/challenge-2020/1.0.2/training/georgia/"
GEORGIA_DIR = "data/georgia"
GEORGIA_GROUPS = [f"g{i}" for i in range(1, 12)]


def _fetch(url, dest, resume=False):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    start = os.path.getsize(dest) if resume and os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if start and r.status != 206:  # server ignored Range -> restart
            start = 0
        with open(dest, "ab" if start else "wb") as f:
            done = start
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if resume:
                    print(f"\r  {done / 1e6:8.0f} MB", end="", flush=True)
    if resume:
        print()


def _ptbxl_one(rel):
    for ext in (".hea", ".dat"):
        dest = os.path.join(PTBXL_DIR, rel + ext)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        for attempt in range(4):
            try:
                _fetch(f"{PTBXL_BASE}{rel}{ext}", dest)
                break
            except Exception as e:  # noqa: BLE001 - retry any network error
                if attempt == 3:
                    return f"{rel}{ext}: {e}"
    return None


def download_ptbxl(workers=16):
    """Per-file fetch of metadata + the 100 Hz records from PhysioNet's open S3 mirror
    (physionet.org itself is often rate-limited; the zip is 1.8 GB and includes 500 Hz)."""
    import pandas as pd
    from tqdm import tqdm
    for f in ("ptbxl_database.csv", "scp_statements.csv"):
        if not os.path.exists(os.path.join(PTBXL_DIR, f)):
            _fetch(PTBXL_BASE + f, os.path.join(PTBXL_DIR, f))
    recs = pd.read_csv(os.path.join(PTBXL_DIR, "ptbxl_database.csv"))["filename_lr"].tolist()
    print(f"{len(recs)} PTB-XL 100 Hz records")
    with ThreadPoolExecutor(workers) as ex:
        errs = [e for e in tqdm(ex.map(_ptbxl_one, recs), total=len(recs)) if e]
    if errs:
        print(f"{len(errs)} failed (re-run to retry), e.g. {errs[:3]}")
        sys.exit(1)
    print("done ->", PTBXL_DIR)


def _georgia_one(args):
    group, rec = args
    for ext in (".hea", ".mat"):
        dest = os.path.join(GEORGIA_DIR, group, rec + ext)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        for attempt in range(3):
            try:
                _fetch(f"{GEORGIA_BASE}{group}/{rec}{ext}", dest)
                break
            except Exception as e:  # noqa: BLE001 - retry any network error
                if attempt == 2:
                    return f"{group}/{rec}{ext}: {e}"
    return None


def download_georgia(workers=32, limit=None, seed=0):
    jobs = []
    for g in GEORGIA_GROUPS:
        for attempt in range(6):
            try:
                with urllib.request.urlopen(f"{GEORGIA_BASE}{g}/RECORDS", timeout=120) as r:
                    jobs += [(g, line.strip()) for line in r.read().decode().splitlines() if line.strip()]
                break
            except Exception:  # noqa: BLE001 - physionet.org is flaky; retry
                if attempt == 5:
                    raise
    if limit and limit < len(jobs):  # seeded random subset across all groups (physionet.org is slow)
        import random
        jobs = sorted(random.Random(seed).sample(jobs, limit))
    print(f"{len(jobs)} Georgia records")
    from tqdm import tqdm
    with ThreadPoolExecutor(workers) as ex:
        errs = [e for e in tqdm(ex.map(_georgia_one, jobs), total=len(jobs)) if e]
    if errs:
        print(f"{len(errs)} failed (re-run to retry), e.g. {errs[:3]}")
        sys.exit(1)
    print("done ->", GEORGIA_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ptbxl", "georgia"])
    ap.add_argument("--limit", type=int, default=None, help="georgia: random subset of N records")
    a = ap.parse_args()
    if a.dataset == "ptbxl":
        download_ptbxl()
    else:
        download_georgia(limit=a.limit)
