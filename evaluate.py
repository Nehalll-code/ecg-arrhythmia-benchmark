"""Shared evaluation harness. All models are scored through these functions."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from data import CLASSES


def best_thresholds(y_true, probs):
    """Per-class threshold maximising F1 on the VALIDATION set."""
    grid = np.linspace(0.05, 0.95, 91)
    th = []
    for c in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, c], probs[:, c] >= t, zero_division=0) for t in grid]
        th.append(grid[int(np.argmax(scores))])
    return np.array(th)


def class_auroc(y_true, probs):
    """Per-class AUROC; NaN where a class has only one label value (undefined)."""
    out = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        out.append(roc_auc_score(col, probs[:, c]) if 0 < col.sum() < len(col) else np.nan)
    return np.array(out)


def macro_auroc(y_true, probs):
    return float(np.nanmean(class_auroc(y_true, probs)))


def evaluate(y_true, probs, thresholds=None):
    """Macro AUROC (primary), macro F1, per-class AUROC/precision/recall/F1.

    Macro averages skip classes with no positives/negatives in `y_true` (NaN AUROC) so the
    same function scores external cohorts that lack a class."""
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    pred = (probs >= thresholds).astype(int)
    auc = class_auroc(y_true, probs)
    present = y_true.sum(0) > 0
    res = {
        "macro_auroc": float(np.nanmean(auc)),
        "macro_f1": float(f1_score(y_true[:, present], pred[:, present], average="macro", zero_division=0)),
        "per_class": {},
        "thresholds": thresholds.tolist(),
    }
    for c, name in enumerate(CLASSES):
        res["per_class"][name] = {
            "n_pos": int(y_true[:, c].sum()),
            "auroc": None if np.isnan(auc[c]) else float(auc[c]),
            "precision": float(precision_score(y_true[:, c], pred[:, c], zero_division=0)),
            "recall": float(recall_score(y_true[:, c], pred[:, c], zero_division=0)),
            "f1": float(f1_score(y_true[:, c], pred[:, c], zero_division=0)),
        }
    return res


def print_report(name, res):
    print(f"\n== {name} ==  macro AUROC {res['macro_auroc']:.4f} | macro F1 {res['macro_f1']:.4f}")
    for c, m in res["per_class"].items():
        auc = "  n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
        print(f"  {c:5s} (n={m['n_pos']:5d}) AUROC {auc}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")


# ------------------------------------------------------------ error analysis
def per_class_confusion(y_true, probs, thresholds):
    """2x2 counts per class: TP, FP, FN, TN."""
    pred = probs >= thresholds
    t = y_true.astype(bool)
    return {name: {"TP": int((pred[:, c] & t[:, c]).sum()), "FP": int((pred[:, c] & ~t[:, c]).sum()),
                   "FN": int((~pred[:, c] & t[:, c]).sum()), "TN": int((~pred[:, c] & ~t[:, c]).sum())}
            for c, name in enumerate(CLASSES)}


def confusion_matrix_multilabel(y_true, probs, thresholds):
    """Row i / col j (j in CLASSES + 'none'): of the records whose TRUE labels include i,
    the fraction predicted j. Diagonal = recall; off-diagonal = which class absorbs
    the misses (e.g. MI recordings being called STTC). 'none' = no class predicted."""
    pred = probs >= thresholds
    t = y_true.astype(bool)
    k = len(CLASSES)
    m = np.zeros((k, k + 1))
    for i in range(k):
        rows = t[:, i]
        if rows.sum() == 0:
            m[i] = np.nan
            continue
        m[i, :k] = pred[rows].mean(0)
        m[i, k] = (~pred[rows].any(1)).mean()
    return m


# ---------------------------------------------------------------- bootstrap
def bootstrap_macro_auroc(y_true, probs_by_model, groups=None, n=1000, seed=0):
    """Resample records (or whole patients if `groups` given) and return, per model, the
    bootstrap macro-AUROC samples. Same resamples for every model -> paired comparisons."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    if groups is None:
        idx_sets = lambda: rng.integers(0, N, N)
    else:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        idx_sets = lambda: np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
    out = {m: [] for m in probs_by_model}
    for _ in range(n):
        idx = idx_sets()
        yt = y_true[idx]
        if (yt.sum(0) == 0).any() or (yt.sum(0) == len(yt)).any():
            continue
        for m, p in probs_by_model.items():
            out[m].append(macro_auroc(yt, p[idx]))
    return {m: np.array(v) for m, v in out.items()}
