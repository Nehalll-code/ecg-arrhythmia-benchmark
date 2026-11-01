"""Fusion extension + complementarity analysis, from saved predictions (no retraining).

    python ensemble.py            # reads outputs/<m>/{val,test}_probs.npy -> results/ensemble.{md,json}, disagreement.png

Fusion methods (everything is fitted on the VALIDATION fold only; the test fold is scored once):
  mean      soft voting, equal weights
  weighted  weights proportional to (val macro AUROC - 0.5) per model
  stacking  per-class logistic regression on the logits of all models (4 x 5 = 20 features),
            fitted on the val fold; val thresholds come from 5-fold out-of-fold predictions so they
            are not fitted on in-sample stacker outputs
Complementarity: pairwise prediction disagreement, error correlation, and oracle coverage.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

from data import CLASSES, SIGNAL_CACHE, load_data, load_test_meta
from evaluate import best_thresholds, bootstrap_macro_auroc, evaluate, macro_auroc

MODELS = ["mlp", "cnn", "bilstm", "transformer"]
LABEL = {"mlp": "MLP", "cnn": "1D-CNN", "bilstm": "BiLSTM", "transformer": "Transformer"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(probs_list):
    return np.concatenate([logit(p) for p in probs_list], axis=1)


def fit_stacker(F, y, C=1.0):
    return [LogisticRegression(C=C, max_iter=2000).fit(F, y[:, c]) for c in range(y.shape[1])]


def predict_stacker(models, F):
    return np.stack([m.predict_proba(F)[:, 1] for m in models], axis=1)


def oof_stacker(F, y, groups, C=1.0, k=5):
    oof = np.zeros(y.shape)
    splitter = GroupKFold(k).split(F, y, groups) if groups is not None else KFold(k, shuffle=True, random_state=42).split(F)
    for tr, te in splitter:
        oof[te] = predict_stacker(fit_stacker(F[tr], y[tr], C), F[te])
    return oof


def main():
    d = load_data()
    yva, yte = d["val"][1], d["test"][1]
    val = {m: np.load(f"outputs/{m}/val_probs.npy") for m in MODELS}
    test = {m: np.load(f"outputs/{m}/test_probs.npy") for m in MODELS}
    for m in MODELS:
        assert val[m].shape == yva.shape and test[m].shape == yte.shape, m

    cache = np.load(SIGNAL_CACHE)
    pid = cache["patient_id"] if "patient_id" in cache else None
    fold = cache["fold"]
    val_groups = pid[fold == 9] if pid is not None else None
    test_groups = load_test_meta()

    # ---- fusion
    val_auc = {m: macro_auroc(yva, val[m]) for m in MODELS}
    w = np.array([val_auc[m] - 0.5 for m in MODELS])
    w = w / w.sum()
    P_val, P_te, notes = {}, {}, {}
    P_val["mean"] = np.mean([val[m] for m in MODELS], 0)
    P_te["mean"] = np.mean([test[m] for m in MODELS], 0)
    P_val["weighted"] = sum(wi * val[m] for wi, m in zip(w, MODELS))
    P_te["weighted"] = sum(wi * test[m] for wi, m in zip(w, MODELS))
    notes["weighted"] = dict(zip(MODELS, np.round(w, 4).tolist()))

    Fva = stack_features([val[m] for m in MODELS])
    Fte = stack_features([test[m] for m in MODELS])
    stackers = fit_stacker(Fva, yva)
    P_te["stacking"] = predict_stacker(stackers, Fte)
    P_val["stacking"] = oof_stacker(Fva, yva, val_groups)  # out-of-fold, used only for thresholds

    singles_te = {m: test[m] for m in MODELS}
    th_single = {m: best_thresholds(yva, val[m]) for m in MODELS}
    res = {}
    for name, p in {**singles_te, **P_te}.items():
        pv = val[name] if name in val else P_val[name]
        th = th_single[name] if name in th_single else best_thresholds(yva, pv)
        res[name] = evaluate(yte, p, th)

    best_single = max(MODELS, key=lambda m: res[m]["macro_auroc"])
    boots = bootstrap_macro_auroc(yte, {**singles_te, **P_te}, groups=test_groups, n=1000, seed=0)
    n = min(len(v) for v in boots.values())
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boots.items()}
    p_beat = {k: float(np.mean(boots[k][:n] > boots[best_single][:n])) for k in P_te}

    # ---- complementarity (single models, val-tuned thresholds, test fold)
    pred = {m: (test[m] >= th_single[m]).astype(int) for m in MODELS}
    err = {m: (pred[m] != yte).astype(int) for m in MODELS}
    k = len(MODELS)
    dis = np.zeros((k, k)); ecorr = np.zeros((k, k)); both_wrong = np.zeros((k, k))
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            dis[i, j] = np.mean(pred[a] != pred[b])
            ecorr[i, j] = np.corrcoef(err[a].ravel(), err[b].ravel())[0, 1]
            both_wrong[i, j] = np.mean((err[a] == 1) & (err[b] == 1)) / max(np.mean(err[a] == 1), 1e-9)
    E = np.stack([err[m] for m in MODELS])
    cells = yte.size
    any_right = float(np.mean(E.min(0) == 0))
    all_wrong = float(np.mean(E.min(0) == 1))
    maj = (np.stack([pred[m] for m in MODELS]).sum(0) >= 2).astype(int)  # hard majority vote
    best_acc = {m: float(1 - err[m].mean()) for m in MODELS}

    # ---- outputs
    os.makedirs("results", exist_ok=True)
    fusion_names = ["mean", "weighted", "stacking"]
    L = {**LABEL, "mean": "Soft vote (mean)", "weighted": "Weighted vote", "stacking": "Stacked LR"}
    lines = ["# Fusion and complementarity (PTB-XL fold-10 test)", "",
             "All fusion parameters are fitted on the validation fold. "
             f"Best single model: **{LABEL[best_single]}**. "
             "'P(beats best single)' is the paired patient-level bootstrap probability that the fused macro AUROC "
             "exceeds the best single model's (about 0.5 = indistinguishable).", "",
             "| System | Macro AUROC [95% CI] | Macro F1 | P(beats best single) |", "|---|---|---|---|"]
    for name in MODELS + fusion_names:
        lo, hi = ci[name]
        pb = f"{p_beat[name]:.2f}" if name in p_beat else "-"
        lines.append(f"| {L[name]} | {res[name]['macro_auroc']:.3f} [{lo:.3f}, {hi:.3f}] | {res[name]['macro_f1']:.3f} | {pb} |")
    lines += ["", f"Weighted-vote weights (from val macro AUROC - 0.5): {notes['weighted']}", "",
              "## Per-class AUROC", "", "| System | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name in MODELS + fusion_names:
        lines.append(f"| {L[name]} | " + " | ".join(f"{res[name]['per_class'][c]['auroc']:.3f}" for c in CLASSES) + " |")
    lines += ["", "## Pairwise prediction disagreement", "",
              "Fraction of (record, class) decisions on which two models differ (binary predictions, val-tuned thresholds).", "",
              "| | " + " | ".join(LABEL[m] for m in MODELS) + " |", "|---|" + "---|" * k]
    for i, a in enumerate(MODELS):
        lines.append(f"| {LABEL[a]} | " + " | ".join(f"{dis[i, j]:.3f}" for j in range(k)) + " |")
    lines += ["", "## Error correlation", "",
              "Pearson correlation of the per-decision error indicators (1 = identical mistakes, 0 = independent).", "",
              "| | " + " | ".join(LABEL[m] for m in MODELS) + " |", "|---|" + "---|" * k]
    for i, a in enumerate(MODELS):
        lines.append(f"| {LABEL[a]} | " + " | ".join(f"{ecorr[i, j]:.2f}" for j in range(k)) + " |")
    lines += ["", "## Oracle coverage", "",
              f"Of {cells:,} (record, class) decisions: all four models wrong on {all_wrong:.1%}; "
              f"at least one model right on {any_right:.1%}. Best single-model decision accuracy is "
              f"{max(best_acc.values()):.1%} ({LABEL[max(best_acc, key=best_acc.get)]}). "
              f"Hard majority vote (>=2 of 4) decision accuracy: {1 - np.mean(maj != yte):.1%}."]
    with open("results/ensemble.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("results/ensemble.json", "w") as f:
        json.dump({"best_single": best_single, "weights": notes["weighted"],
                   "macro_auroc": {k_: res[k_]["macro_auroc"] for k_ in res},
                   "macro_f1": {k_: res[k_]["macro_f1"] for k_ in res},
                   "ci95": ci, "p_beats_best_single": p_beat,
                   "disagreement": dis.tolist(), "error_correlation": ecorr.tolist(),
                   "all_wrong": all_wrong, "any_right": any_right, "models": MODELS}, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a_, M, t, fm in [(ax[0], dis, "Prediction disagreement", "{:.3f}"), (ax[1], ecorr, "Error correlation", "{:.2f}")]:
        im = a_.imshow(M, cmap="viridis")
        a_.set_xticks(range(k)); a_.set_yticks(range(k))
        a_.set_xticklabels([LABEL[m] for m in MODELS], rotation=30, ha="right"); a_.set_yticklabels([LABEL[m] for m in MODELS])
        for i in range(k):
            for j in range(k):
                a_.text(j, i, fm.format(M[i, j]), ha="center", va="center", color="white", fontsize=8)
        a_.set_title(t); fig.colorbar(im, ax=a_, fraction=0.046)
    fig.tight_layout(); fig.savefig("results/disagreement.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
