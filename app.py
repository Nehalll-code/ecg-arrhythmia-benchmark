"""Streamlit demo: four architectures, one ECG, plus the benchmark graphs.

    python make_demo_bundle.py       # once, after training
    streamlit run app.py
Reads only from demo/ (override with ECG_DEMO_DIR) so it can be deployed without the 1 GB cache.
Research demo, not a diagnostic tool.
"""
import json
import os
import time

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from data import CLASSES, _record_features
from models import build_model, input_kind

DEMO = os.environ.get("ECG_DEMO_DIR", "demo")
MODELS = ["mlp", "cnn", "bilstm", "transformer"]
LABEL = {"mlp": "MLP", "cnn": "1D-CNN", "bilstm": "BiLSTM", "transformer": "Transformer"}
FULL = {"NORM": "Normal", "MI": "Myocardial infarction", "STTC": "ST/T change",
        "CD": "Conduction disturbance", "HYP": "Hypertrophy"}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

st.set_page_config(page_title="ECG architecture benchmark", layout="wide")
torch.set_num_threads(1)  # same single-thread CPU setting as efficiency.py


@st.cache_resource
def load_models():
    out = {}
    for m in MODELS:
        d = os.path.join(DEMO, "models", m)
        if not os.path.exists(os.path.join(d, "best.pt")):
            continue
        meta = json.load(open(os.path.join(d, "metrics.json")))
        net = build_model(m, **meta["model_kw"])
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu"))
        norm = np.load(os.path.join(d, "feat_norm.npz")) if m == "mlp" else None
        out[m] = {"net": net.eval(), "th": np.array(meta["val"]["thresholds"]),
                  "norm": None if norm is None else (norm["mu"], norm["sd"])}
    return out


@st.cache_data
def load_sample():
    p = os.path.join(DEMO, "sample.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


@st.cache_data
def load_json(name):
    p = os.path.join(DEMO, name)
    return json.load(open(p)) if os.path.exists(p) else None


def predict(m, spec, sig):
    """-> (probs[5], end-to-end ms). The MLP's time includes hand-crafted feature extraction."""
    t0 = time.perf_counter()
    with torch.no_grad():
        if input_kind(m) == "signal":
            x = torch.from_numpy(sig[None])
        else:
            mu, sd = spec["norm"]
            x = torch.from_numpy(((_record_features(sig) - mu) / sd).astype(np.float32)[None])
        p = torch.sigmoid(spec["net"](x))[0].numpy()
    return p, (time.perf_counter() - t0) * 1e3


def plot_ecg(sig):
    fig, axes = plt.subplots(6, 2, figsize=(11, 6.2), sharex=True)
    t = np.arange(len(sig)) / 100
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(t, sig[:, i], lw=0.7, color="#c0392b")
        ax.set_ylabel(LEADS[i], rotation=0, labelpad=12, fontsize=8)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    fig.tight_layout()
    return fig


def tab_try(models, sample):
    if not models or sample is None:
        st.warning("No trained models / sample found. Run `python run_all.py` then `python make_demo_bundle.py`.")
        return
    srcs = [s for s in ("ptbxl", "georgia") if f"X_{s}" in sample]
    c1, c2, c3 = st.columns([1.2, 2, 1])
    src = c1.radio("ECG source", srcs, format_func=lambda s: {
        "ptbxl": "PTB-XL test fold (in-distribution)", "georgia": "Georgia (unseen hospital)"}[s])
    n = len(sample[f"X_{src}"])
    if "i" not in st.session_state:
        st.session_state.i = 0
    if c3.button("Random ECG"):
        st.session_state.i = int(np.random.randint(n))
    i = c2.slider("Record", 0, n - 1, min(st.session_state.i, n - 1), key="slider_i")
    st.session_state.i = i
    sig, y = sample[f"X_{src}"][i], sample[f"y_{src}"][i]
    truth = [c for c, v in zip(CLASSES, y) if v]
    st.markdown("**True diagnosis:** " + (", ".join(f"`{c}` {FULL[c]}" for c in truth) or "-"))
    if src == "georgia":
        st.caption("MI is not scored on Georgia (too few cases), so MI outputs are not meaningful there.")
    left, right = st.columns([3, 2])
    left.pyplot(plot_ecg(sig), clear_figure=True)

    rows, lat, pred = [], {}, {}
    for m, spec in models.items():
        p, lat[m] = predict(m, spec, sig)
        pred[m] = p >= spec["th"]
        for c, pc, th in zip(CLASSES, p, spec["th"]):
            rows.append({"Model": LABEL[m], "Class": c, "Probability": float(pc), "Threshold": float(th),
                         "Predicted": bool(pc >= th)})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Class:N", sort=CLASSES, title=None), xOffset="Model:N",
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", sort=[LABEL[m] for m in MODELS]),
        tooltip=["Model", "Class", alt.Tooltip("Probability:Q", format=".2f"), "Predicted"]
    ).properties(height=260)
    ticks = alt.Chart(df).mark_tick(color="black", thickness=2, size=14).encode(
        x=alt.X("Class:N", sort=CLASSES), xOffset="Model:N", y="Threshold:Q")
    right.altair_chart(chart + ticks, width="stretch")
    right.caption("Black ticks = each model's decision threshold (tuned on the validation fold).")

    table = pd.DataFrame({LABEL[m]: ["✔" if v else "·" for v in pred[m]] for m in models}, index=CLASSES).T
    table["Latency (ms)"] = [f"{lat[m]:.1f}" for m in models]
    st.dataframe(table, width="stretch")
    ok = [LABEL[m] for m in models if (pred[m] == y.astype(bool)).all()]
    st.write(f"**Exactly right (all 5 labels):** {', '.join(ok) or 'none of the models'}.")
    if "mlp" in models:
        st.caption("MLP latency includes hand-crafted feature extraction, which dominates its cost.")


def tab_results(models):
    summ, eff = load_json("results/summary.json"), load_json("efficiency.json")
    if not summ:
        st.info("No results found. Run `python run_all.py` and `python make_demo_bundle.py`.")
        return
    names = [m for m in MODELS if m in summ]
    rows = [{"Model": LABEL[m], "Class": c, "AUROC": summ[m]["in_dist"]["per_class"][c]["auroc"]}
            for m in names for c in CLASSES]
    st.subheader("Per-class AUROC, PTB-XL test")
    st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("Class:N", sort=CLASSES, title=None), xOffset="Model:N",
        y=alt.Y("AUROC:Q", scale=alt.Scale(domain=[0.5, 1])), color="Model:N",
        tooltip=["Model", "Class", alt.Tooltip("AUROC:Q", format=".3f")]).properties(height=300),
        width="stretch")

    ext = [m for m in names if "external" in summ[m]]
    if ext:
        st.subheader("Does the winner survive a new hospital?")
        cls = "/".join(summ[ext[0]]["external"]["classes"])
        rows = [{"Model": LABEL[m], "Dataset": d, "Macro AUROC": summ[m]["external"][k]}
                for m in ext for d, k in (("PTB-XL test", "ptbxl_same_classes"), ("Georgia (external)", "macro_auroc_scored"))]
        st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
            x=alt.X("Model:N", title=None), xOffset="Dataset:N",
            y=alt.Y("Macro AUROC:Q", scale=alt.Scale(domain=[0.5, 1])), color="Dataset:N",
            tooltip=["Model", "Dataset", alt.Tooltip("Macro AUROC:Q", format=".3f")]).properties(height=300),
            width="stretch")
        st.caption(f"Both bars average the same classes ({cls}) so the drop is like-for-like.")

    if eff:
        st.subheader("Accuracy vs cost")
        rows = [{"Model": LABEL[m], "Macro AUROC": summ[m]["in_dist"]["macro_auroc"],
                 "Latency (ms)": eff[m]["latency_ms_end_to_end"], "Params": eff[m]["params"],
                 "MFLOPs": eff[m]["flops"] / 1e6} for m in names]
        st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_circle(size=250).encode(
            x=alt.X("Latency (ms):Q"), y=alt.Y("Macro AUROC:Q", scale=alt.Scale(zero=False)),
            color="Model:N", tooltip=["Model", "Macro AUROC", "Latency (ms)", "Params", "MFLOPs"]
        ).properties(height=300), width="stretch")
        st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    st.subheader("Error analysis")
    for f, cap in (("confusion.png", "Which class absorbs a model's misses (rows = true class)"),
                   ("roc.png", "ROC curves per class"), ("training_curves.png", "Validation AUROC while training")):
        p = os.path.join(DEMO, "results", f)
        if os.path.exists(p):
            st.image(p, caption=cap, width="stretch")
    md = os.path.join(DEMO, "results", "results.md")
    if os.path.exists(md):
        with st.expander("Full tables (confidence intervals, paired tests, per-class precision/recall)"):
            st.markdown(open(md, encoding="utf-8").read())


def tab_about():
    st.markdown("""
**Question:** four different neural-network families read the *same* PTB-XL ECGs under the *same* protocol
(patient-disjoint folds 1-8 / 9 / 10, BCE with sigmoid outputs, AdamW, seed 42, early stopping on validation macro AUROC).
Which wins, does it stay the winner on an unseen hospital (Georgia), and what does it cost?

| Model | Assumes the diagnostic information is in... |
|---|---|
| MLP | hand-engineered features (RR/HRV, QRS width, band powers); never sees the raw waveform |
| 1D-CNN | local waveform shape of each beat |
| BiLSTM | beat-to-beat temporal context, both directions |
| Transformer | global relations between any two segments of the 10 s recording |

**Read the numbers carefully:** accuracy is never reported (a model that always says "normal" would look good); we use macro AUROC with
bootstrap confidence intervals, and only call one model better when a paired bootstrap supports it.

**This is a research demo, not a diagnostic tool.** Models are trained on one dataset and degrade on another hospital's data;
Georgia has almost no MI cases, so MI is not scored externally.
""")


st.title("Four architectures, one ECG")
models = load_models()
sample = load_sample()
t1, t2, t3 = st.tabs(["Try an ECG", "Benchmark graphs", "About"])
with t1:
    tab_try(models, sample)
with t2:
    tab_results(models)
with t3:
    tab_about()
