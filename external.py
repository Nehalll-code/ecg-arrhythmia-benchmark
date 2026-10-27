"""External cohort: Georgia 12-lead ECG database (PhysioNet/CinC Challenge 2020), mapped onto the
same 5 PTB-XL superclasses and pushed through the SAME preprocessing (100 Hz, 10 s, 0.5 Hz
high-pass, PTB-XL train-fold z-score).

    python external.py --data-dir data/georgia            # -> cache/georgia_100hz.npz, cache/georgia_features.npz

Label mapping (SNOMED CT -> superclass) follows the PTB-XL SCP-statement superclass definitions
(verified against scp_statements.csv): ischaemia / long-QT / non-specific ST-T -> STTC,
all AV blocks and bundle-branch / fascicular blocks -> CD, atrial enlargement counts as HYP.
As in PTB-XL we drop records with no superclass (e.g. rhythm-only: AF, bradycardia).
NORM = sinus rhythm with no other mapped abnormality. MI is almost absent in Georgia
(~7 records of 10k), so MI is reported as not evaluable externally.
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy.signal import resample_poly

from data import CLASSES, FS, load_norm_stats, preprocess_signal, _record_features

EXT_SIGNAL_CACHE = "cache/georgia_100hz.npz"
EXT_FEATURE_CACHE = "cache/georgia_features.npz"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # PTB-XL order

SNOMED_TO_CLASS = {
    # MI
    "164865005": "MI", "164867002": "MI", "57054005": "MI", "54329005": "MI",
    # STTC: ST/T changes, ischaemia, long QT
    "429622005": "STTC", "164931005": "STTC", "164934002": "STTC", "59931005": "STTC",
    "55930002": "STTC", "428750005": "STTC", "164930006": "STTC", "704997005": "STTC",
    "426434006": "STTC", "425419005": "STTC", "425623009": "STTC", "164861001": "STTC",
    "413444003": "STTC", "413844008": "STTC", "111975006": "STTC", "370365005": "STTC",
    # CD: bundle-branch / fascicular / AV blocks / IVCD
    "270492004": "CD", "195042002": "CD", "27885002": "CD", "233917008": "CD",
    "713427006": "CD", "59118001": "CD", "713426002": "CD", "164909002": "CD",
    "251120003": "CD", "445118002": "CD", "445211001": "CD", "698252002": "CD", "6374002": "CD",
    # HYP: ventricular hypertrophy, atrial enlargement
    "164873001": "HYP", "89792004": "HYP", "266249003": "HYP", "67741000119109": "HYP",
    "253352002": "HYP", "253339007": "HYP", "195126007": "HYP", "446358003": "HYP", "446813000": "HYP",
}
NSR = "426783006"


def map_labels(snomed_codes):
    """List of SNOMED codes -> 5-dim multi-hot (all-zero if the record has no superclass)."""
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for c in snomed_codes:
        if c in SNOMED_TO_CLASS:
            y[CLASSES.index(SNOMED_TO_CLASS[c])] = 1.0
    if y.sum() == 0 and NSR in snomed_codes:
        y[CLASSES.index("NORM")] = 1.0
    return y


def read_dx(hea_path):
    with open(hea_path) as f:
        m = re.search(r"^#\s*Dx:\s*(.*)$", f.read(), re.M)
    return [c.strip() for c in m.group(1).split(",") if c.strip()] if m else []


def to_100hz(sig, fs, names):
    """(T, n_leads) at `fs` -> (T', 12) at 100 Hz in PTB-XL lead order, or None if leads missing."""
    idx = {n.upper(): i for i, n in enumerate(names)}
    if not all(l in idx for l in LEADS):
        return None
    sig = sig[:, [idx[l] for l in LEADS]]
    if fs != FS:
        sig = resample_poly(np.nan_to_num(sig), FS, int(fs), axis=0)
    return sig


def fix_length(x, n=1000):
    """Crop to the first 10 s or zero-pad (post-normalisation zero = lead mean)."""
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), x.shape[1]), dtype=x.dtype)])


def build_external_cache(data_dir, features=True, min_seconds=5):
    import wfdb
    from tqdm import tqdm
    mu, sd = load_norm_stats()
    heas = sorted(glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True))
    X, Y, ids, skipped = [], [], [], {"no_superclass": 0, "bad_signal": 0}
    for h in tqdm(heas):
        base = h[:-4]
        y = map_labels(read_dx(h))
        if y.sum() == 0:
            skipped["no_superclass"] += 1
            continue
        try:
            sig, meta = wfdb.rdsamp(base)
            sig = to_100hz(sig, meta["fs"], meta["sig_name"])
        except Exception:  # noqa: BLE001 - corrupt/partial download
            sig = None
        if sig is None or len(sig) < min_seconds * FS:
            skipped["bad_signal"] += 1
            continue
        X.append(fix_length(preprocess_signal(sig, mu, sd)))
        Y.append(y)
        ids.append(os.path.basename(base))
    X, Y = np.stack(X), np.stack(Y)
    os.makedirs("cache", exist_ok=True)
    np.savez_compressed(EXT_SIGNAL_CACHE, X=X, y=Y, ids=np.array(ids))
    print(f"saved {EXT_SIGNAL_CACHE} {X.shape}; skipped {skipped}")
    print("label support (n positives):", dict(zip(CLASSES, Y.sum(0).astype(int))))
    if features:
        F = np.stack([_record_features(x) for x in tqdm(X, desc="features")])
        np.savez_compressed(EXT_FEATURE_CACHE, F=F, y=Y)
        print("saved", EXT_FEATURE_CACHE, F.shape)


def load_external(path=EXT_SIGNAL_CACHE):
    d = np.load(path)
    return (d["X"] if "X" in d else d["F"]), d["y"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/georgia")
    ap.add_argument("--no-features", action="store_true")
    a = ap.parse_args()
    build_external_cache(a.data_dir, features=not a.no_features)
