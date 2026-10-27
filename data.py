"""Shared PTB-XL preprocessing. Every model consumes the tensors produced here.

Usage:
    python data.py --data-dir data/ptb-xl-1.0.3        # builds cache/ptbxl_100hz.npz
    python data.py --data-dir ... --features           # also builds cache/features.npz (MLP)

Splits (official, patient-disjoint): folds 1-8 train, 9 val, 10 test.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FS = 100
CACHE_DIR = "cache"
SIGNAL_CACHE = os.path.join(CACHE_DIR, "ptbxl_100hz.npz")
FEATURE_CACHE = os.path.join(CACHE_DIR, "features.npz")


def _aggregate(scp_codes, agg_df):
    """Map a record's SCP statements to the set of diagnostic superclasses."""
    out = set()
    for code in scp_codes:
        if code in agg_df.index:
            out.add(agg_df.loc[code, "diagnostic_class"])
    return sorted(out)


def _remove_baseline(x, fs=FS, cutoff=0.5):
    sos = butter(2, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def build_signal_cache(data_dir, baseline_removal=True):
    import wfdb
    from tqdm import tqdm

    meta = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    agg = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    agg = agg[agg["diagnostic"] == 1]

    meta["superclasses"] = meta["scp_codes"].apply(lambda c: _aggregate(c, agg))
    meta = meta[meta["superclasses"].apply(len) > 0]  # drop records with no superclass
    print(f"{len(meta)} records with >=1 superclass")

    X = np.zeros((len(meta), 1000, 12), dtype=np.float32)
    y = np.zeros((len(meta), len(CLASSES)), dtype=np.float32)
    for i, (ecg_id, row) in enumerate(tqdm(meta.iterrows(), total=len(meta))):
        sig, _ = wfdb.rdsamp(os.path.join(data_dir, row["filename_lr"]))
        if baseline_removal:
            sig = _remove_baseline(sig)
        X[i] = sig
        for c in row["superclasses"]:
            y[i, CLASSES.index(c)] = 1.0

    fold = meta["strat_fold"].values.astype(np.int64)
    # per-lead z-score using TRAIN-fold statistics only (no val/test leakage)
    tr = np.flatnonzero(fold <= 8)
    # streamed in float64 and normalised in place: avoids ~2 extra copies of the 1 GB array
    n = len(tr) * X.shape[1]
    s1 = sum(X[i].sum(0, dtype=np.float64) for i in tr)
    mu = (s1 / n).astype(np.float32)
    s2 = sum(((X[i] - mu) ** 2).sum(0, dtype=np.float64) for i in tr)
    sd = (np.sqrt(s2 / n) + 1e-8).astype(np.float32)
    mu, sd = mu[None, None], sd[None, None]  # (1, 1, 12)
    X -= mu
    X /= sd

    os.makedirs(CACHE_DIR, exist_ok=True)
    # mu/sd are stored so external cohorts are normalised with the SAME PTB-XL statistics
    np.savez_compressed(SIGNAL_CACHE, X=X, y=y, fold=fold, ecg_id=meta.index.values,
                        patient_id=meta["patient_id"].values.astype(np.int64),
                        mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    print("saved", SIGNAL_CACHE, X.shape, y.shape, "label prevalence:",
          dict(zip(CLASSES, y.mean(0).round(3))))


def load_norm_stats(path=SIGNAL_CACHE):
    """PTB-XL train-fold per-lead (mu, sd), shape (1, 1, 12). Needs a cache built by this version."""
    d = np.load(path)
    if "mu" not in d:
        raise KeyError(f"{path} has no mu/sd; rebuild it with `python data.py --data-dir ...`")
    return d["mu"], d["sd"]


def preprocess_signal(sig, mu, sd, fs=FS):
    """Raw (T, 12) float signal already at `fs` -> baseline-removed, PTB-XL-normalised (1000, 12)."""
    sig = np.nan_to_num(np.asarray(sig, dtype=np.float64))
    sig = _remove_baseline(sig, fs=fs)
    return ((sig - mu[0]) / sd[0]).astype(np.float32)


def load_data(path=SIGNAL_CACHE):
    """Returns dict with train/val/test splits: {'train': (X, y), ...} plus 'fold'."""
    d = np.load(path)
    X, y, fold = d["X"], d["y"], d["fold"]
    return {
        "train": (X[fold <= 8], y[fold <= 8]),
        "val": (X[fold == 9], y[fold == 9]),
        "test": (X[fold == 10], y[fold == 10]),
    }


# ----------------------------------------------------------------- features
def _record_features(sig, fs=FS):
    """Engineered features for one (1000, 12) record: rhythm/HRV, QRS width,
    per-lead statistics and frequency-band powers."""
    from scipy.signal import find_peaks, peak_widths, welch
    from scipy.stats import kurtosis, skew

    f = []
    # --- R-peaks on lead II (index 1): RR stats, HRV, QRS width
    lead = sig[:, 1]
    peaks, props = find_peaks(lead, distance=int(0.3 * fs),
                              height=np.mean(lead) + 1.0 * np.std(lead), prominence=0.5)
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        drr = np.diff(rr)
        widths = peak_widths(lead, peaks, rel_height=0.5)[0] / fs
        f += [len(peaks), 60.0 / rr.mean(), rr.mean(), rr.std(), rr.min(), rr.max(),
              np.sqrt(np.mean(drr ** 2)) if len(drr) else 0.0,            # RMSSD
              np.mean(np.abs(drr) > 0.05) if len(drr) else 0.0,          # pNN50-like
              np.median(widths), props["peak_heights"].mean()]
    else:
        f += [len(peaks)] + [0.0] * 9
    # --- per-lead statistics + spectral band powers
    freqs, psd = welch(sig, fs=fs, nperseg=256, axis=0)
    bands = [(0.5, 4), (4, 10), (10, 20), (20, 40)]
    total = psd.sum(0) + 1e-12
    for l in range(sig.shape[1]):
        s = sig[:, l]
        f += [s.mean(), s.std(), s.min(), s.max(), np.ptp(s), skew(s), kurtosis(s),
              np.mean(np.abs(np.diff(s)))]
        for lo, hi in bands:
            m = (freqs >= lo) & (freqs < hi)
            f.append(psd[m, l].sum() / total[l])
    return np.nan_to_num(np.array(f, dtype=np.float32))


def build_feature_cache():
    from tqdm import tqdm
    d = np.load(SIGNAL_CACHE)
    F = np.stack([_record_features(x) for x in tqdm(d["X"])])
    np.savez_compressed(FEATURE_CACHE, F=F, y=d["y"], fold=d["fold"])
    print("saved", FEATURE_CACHE, F.shape)


def load_test_meta(path=SIGNAL_CACHE):
    """patient_id of the test fold (for patient-level bootstrap); None for old caches."""
    d = np.load(path)
    return d["patient_id"][d["fold"] == 10] if "patient_id" in d else None


def load_features(path=FEATURE_CACHE):
    d = np.load(path)
    F, y, fold = d["F"], d["y"], d["fold"]
    return {
        "train": (F[fold <= 8], y[fold <= 8]),
        "val": (F[fold == 9], y[fold == 9]),
        "test": (F[fold == 10], y[fold == 10]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="PTB-XL root (contains ptbxl_database.csv)")
    ap.add_argument("--no-baseline-removal", action="store_true")
    ap.add_argument("--features", action="store_true", help="also build MLP feature cache")
    a = ap.parse_args()
    build_signal_cache(a.data_dir, baseline_removal=not a.no_baseline_removal)
    if a.features:
        build_feature_cache()
