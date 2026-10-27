"""Build the comparison tables / figures from the per-model outputs.

    python compare.py            # reads outputs/, writes results/{results.md, *.png, summary.json}
In-distribution: PTB-XL fold-10 test. External: Georgia (classes with >= MIN_POS positives only).
CIs: patient-level bootstrap (test) / record-level bootstrap (Georgia). Model-vs-model
significance: paired bootstrap on identical resamples. 'ensemble' = mean of the models' probs.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from data import CLASSES, load_data, load_test_meta
from evaluate import (best_thresholds, bootstrap_macro_auroc, class_auroc,
                      confusion_matrix_multilabel, evaluate, per_class_confusion)
from external import EXT_SIGNAL_CACHE, load_external

MODELS = ["mlp", "cnn", "bilstm", "transformer"]
LABEL = {"mlp": "MLP", "cnn": "1D-CNN", "bilstm": "BiLSTM", "transformer": "Transformer",
         "ensemble": "Ensemble (mean)"}
MIN_POS = 30  # a class needs this many external positives (and negatives) to be scored


def fmt_ci(point, boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def sub_macro(y, p, cols):
    return float(np.nanmean(class_auroc(y, p)[cols]))


def bootstrap_subset(y, probs, cols, groups=None, n=1000, seed=0):
    return bootstrap_macro_auroc(y[:, cols], {m: p[:, cols] for m, p in probs.items()},
                                 groups=groups, n=n, seed=seed)


def pairwise_table(boots, names):
    """P(row model's bootstrap macro AUROC > column model's), paired resamples."""
    lines = ["| | " + " | ".join(LABEL[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        row = []
        for b in names:
            k = min(len(boots[a]), len(boots[b]))
            row.append("-" if a == b else f"{np.mean(boots[a][:k] > boots[b][:k]):.2f}")
        lines.append(f"| {LABEL[a]} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)

    names = [m for m in MODELS if os.path.exists(os.path.join(a.out, m, "test_probs.npy"))]
    if not names:
        raise SystemExit("no model outputs found")
    d = load_data()
    yva, yte = d["val"][1], d["test"][1]
    groups = load_test_meta()
    P = {m: np.load(os.path.join(a.out, m, "test_probs.npy")) for m in names}
    VAL_P = {m: np.load(os.path.join(a.out, m, "val_probs.npy")) for m in names}
    TH = {m: np.array(json.load(open(os.path.join(a.out, m, "metrics.json")))["val"]["thresholds"])
          for m in names}
    if len(names) > 1:  # ensemble: mean probs, thresholds re-fit on val like every other model
        P["ensemble"] = np.mean([P[m] for m in names], 0)
        VAL_P["ensemble"] = np.mean([VAL_P[m] for m in names], 0)
        TH["ensemble"] = best_thresholds(yva, VAL_P["ensemble"])
    all_names = list(P)

    md = ["# Results\n"]
    summary = {}

    # ---------------- in-distribution ----------------
    boots = bootstrap_subset(yte, P, list(range(5)), groups, a.n_boot)
    md += ["## 1. PTB-XL test (fold 10), in-distribution\n",
           "| Model | Macro AUROC [95% CI] | Macro F1 | " + " | ".join(f"AUROC {c}" for c in CLASSES) + " |",
           "|---|---|---|" + "---|" * len(CLASSES)]
    for m in all_names:
        r = evaluate(yte, P[m], TH[m])
        summary[m] = {"in_dist": r}
        md.append(f"| {LABEL[m]} | {fmt_ci(r['macro_auroc'], boots[m])} | {r['macro_f1']:.3f} | "
                  + " | ".join(f"{r['per_class'][c]['auroc']:.3f}" for c in CLASSES) + " |")
    md += ["\nP(row > column), paired patient-level bootstrap of macro AUROC "
           "(~0.5 = indistinguishable; >0.95 or <0.05 = clear difference):\n",
           pairwise_table(boots, all_names), ""]

    md += ["## 2. Per-class precision / recall / F1 (thresholds tuned on val fold)\n",
           "| Model | Class | n+ | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for m in all_names:
        for c in CLASSES:
            pc = summary[m]["in_dist"]["per_class"][c]
            md.append(f"| {LABEL[m]} | {c} | {pc['n_pos']} | {pc['precision']:.3f} | "
                      f"{pc['recall']:.3f} | {pc['f1']:.3f} |")
    md.append("")

    # ---------------- external ----------------
    ext_names = [m for m in names if os.path.exists(os.path.join(a.out, m, "ext_probs.npy"))]
    if ext_names and os.path.exists(EXT_SIGNAL_CACHE):
        _, yext = load_external()
        EP = {m: np.load(os.path.join(a.out, m, "ext_probs.npy")) for m in ext_names}
        if len(ext_names) > 1:
            EP["ensemble"] = np.mean([EP[m] for m in ext_names], 0)
        cols = [c for c in range(5)
                if yext[:, c].sum() >= MIN_POS and (1 - yext[:, c]).sum() >= MIN_POS]
        cn = [CLASSES[c] for c in cols]
        dropped = ", ".join(f"{CLASSES[c]}: n+={int(yext[:, c].sum())}" for c in range(5) if c not in cols)
        eb = bootstrap_subset(yext, EP, cols, None, a.n_boot)
        ib = bootstrap_subset(yte, {m: P[m] for m in EP}, cols, groups, a.n_boot)
        md += [f"## 3. External validation: Georgia (n={len(yext)}), zero-shot, no re-tuning\n",
               f"Scored classes (>= {MIN_POS} positives): {', '.join(cn)}. Not scored: {dropped or 'none'}.\n",
               "Both macro AUROCs average over the SAME classes, so the drop is like-for-like.\n",
               "| Model | PTB-XL test | Georgia | Delta | Georgia rank | PTB-XL rank | "
               + " | ".join(cn) + " |",
               "|---|---|---|---|---|---|" + "---|" * len(cn)]
        s_in = {m: sub_macro(yte, P[m], cols) for m in EP}
        s_ex = {m: sub_macro(yext, EP[m], cols) for m in EP}

        def rank(dct):
            return {m: i + 1 for i, m in enumerate(sorted(dct, key=dct.get, reverse=True))}
        r_in, r_ex = rank(s_in), rank(s_ex)
        for m in EP:
            pcs = class_auroc(yext, EP[m])
            md.append(f"| {LABEL[m]} | {fmt_ci(s_in[m], ib[m])} | {fmt_ci(s_ex[m], eb[m])} | "
                      f"{s_ex[m] - s_in[m]:+.3f} | {r_ex[m]} | {r_in[m]} | "
                      + " | ".join(f"{pcs[c]:.3f}" for c in cols) + " |")
            summary[m]["external"] = {"macro_auroc_scored": s_ex[m], "ptbxl_same_classes": s_in[m],
                                      "classes": cn, "rank_ext": r_ex[m], "rank_ptbxl": r_in[m]}
        md += ["\nP(row > column) on Georgia (paired record-level bootstrap):\n",
               pairwise_table(eb, list(EP)), ""]
        singles = [m for m in EP if m != "ensemble"]
        best_in = max(singles, key=lambda m: s_in[m])
        best_ex = max(singles, key=lambda m: s_ex[m])
        md.append(f"**Winner in-distribution (same classes): {LABEL[best_in]}; on Georgia: {LABEL[best_ex]}.** "
                  f"The winner {'persists' if best_in == best_ex else 'CHANGES'} under distribution shift.\n")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        xs = np.arange(len(EP))
        ax.bar(xs - 0.2, [s_in[m] for m in EP], 0.4, label="PTB-XL test")
        ax.bar(xs + 0.2, [s_ex[m] for m in EP], 0.4, label="Georgia (external)")
        ax.set_xticks(xs, [LABEL[m] for m in EP], fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("macro AUROC (" + "/".join(cn) + ")")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.results, "external_vs_internal.png"), dpi=150)
        plt.close(fig)

    # ---------------- efficiency ----------------
    eff_path = os.path.join(a.out, "efficiency.json")
    if os.path.exists(eff_path):
        eff = json.load(open(eff_path))
        md += ["## 4. Efficiency (per ECG; CPU, 1 thread, batch 1)\n",
               "| Model | Params | MFLOPs | Latency model-only (ms) | Latency end-to-end (ms) | Macro AUROC |",
               "|---|---|---|---|---|---|"]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for m in names:
            e = eff[m]
            auc = summary[m]["in_dist"]["macro_auroc"]
            md.append(f"| {LABEL[m]} | {e['params']:,} | {e['flops'] / 1e6:.1f} | {e['latency_ms_model']:.2f} | "
                      f"{e['latency_ms_end_to_end']:.2f} | {auc:.3f} |")
            ax.scatter(e["latency_ms_end_to_end"], auc, s=40)
            ax.annotate(LABEL[m], (e["latency_ms_end_to_end"], auc),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
            summary[m]["efficiency"] = e
        md.append(f"\nMLP end-to-end includes hand-crafted feature extraction "
                  f"({eff['_feature_extraction_ms']:.1f} ms/ECG).\n")
        ax.set_xlabel("CPU latency per ECG, end-to-end (ms)")
        ax.set_ylabel("PTB-XL test macro AUROC")
        fig.tight_layout()
        fig.savefig(os.path.join(a.results, "accuracy_vs_latency.png"), dpi=150)
        plt.close(fig)

    # ---------------- error analysis ----------------
    md += ["## 5. Error analysis (PTB-XL test)\n",
           "Rows = true class; columns = what the model predicted for those records (multi-label, so a row "
           "can sum to >1). Diagonal = recall; off-diagonal = confusion with another class; "
           "'none' = nothing predicted.\n"]
    fig, axes = plt.subplots(1, len(all_names), figsize=(3.4 * len(all_names), 3.4), squeeze=False)
    for ax, m in zip(axes[0], all_names):
        cm = confusion_matrix_multilabel(yte, P[m], TH[m])
        ax.imshow(np.nan_to_num(cm), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(6), CLASSES + ["none"], fontsize=7, rotation=45)
        ax.set_yticks(range(5), CLASSES, fontsize=7)
        ax.set_title(LABEL[m], fontsize=9)
        for i in range(5):
            for j in range(6):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.6 else "black")
        md += [f"**{LABEL[m]}**\n", "| true \\ predicted | " + " | ".join(CLASSES + ["none"]) + " |",
               "|---|" + "---|" * 6]
        md += [f"| {CLASSES[i]} | " + " | ".join(f"{cm[i, j]:.2f}" for j in range(6)) + " |"
               for i in range(5)]
        pcc = per_class_confusion(yte, P[m], TH[m])
        md.append("\nTP/FP/FN/TN: " + "; ".join(
            f"{c}: {v['TP']}/{v['FP']}/{v['FN']}/{v['TN']}" for c, v in pcc.items()) + "\n")
        summary[m]["confusion_matrix"] = cm.tolist()
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "confusion.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
    for c, ax in enumerate(axes):
        for m in all_names:
            fpr, tpr, _ = roc_curve(yte[:, c], P[m][:, c])
            ax.plot(fpr, tpr, label=LABEL[m], lw=1)
        ax.plot([0, 1], [0, 1], "k:", lw=0.5)
        ax.set_title(CLASSES[c], fontsize=9)
        ax.set_xlabel("FPR", fontsize=8)
    axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "roc.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for m in names:
        h = json.load(open(os.path.join(a.out, m, "metrics.json")))["history"]
        ax.plot([e["epoch"] for e in h], [e["val_auroc"] for e in h], label=LABEL[m])
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro AUROC")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "training_curves.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(a.results, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(a.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    main()
