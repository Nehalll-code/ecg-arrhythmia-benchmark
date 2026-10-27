# Multi-label ECG diagnosis on PTB-XL: MLP vs 1D-CNN vs BiLSTM vs Transformer

We benchmark four architecture families on PTB-XL under a common protocol, then evaluate
zero-shot generalisation to an external cohort (Georgia) and characterise the accuracy-efficiency
trade-off, asking whether architectural advantages observed in-distribution persist under
distribution shift.

Identical data, split (official patient-disjoint folds 1-8 train / 9 val / 10 test), loss
(BCE-with-logits, sigmoid outputs, multi-label), optimiser (AdamW), seed (42) and early stopping
(val macro AUROC) for every model. **Only the model class differs.**

| Model | Input | File | Owner |
|---|---|---|---|
| MLP | engineered features (RR/HRV, QRS width, per-lead stats, band powers) | `models/mlp.py` | M1 |
| 1D-CNN (ResNet) | raw 1000x12 | `models/cnn.py` | M2 |
| BiLSTM | raw 1000x12 | `models/bilstm.py` | M3 |
| Transformer encoder | raw 1000x12 (patch-embedded) | `models/transformer.py` | M4 |

## Pipeline
    pip install -r requirements.txt
    python download.py ptbxl                       # metadata + 100 Hz records (~0.3 GB, S3 mirror) -> data/
    python download.py georgia --limit 3000        # external cohort (seeded random subset) -> data/
    python data.py --data-dir data/ptb-xl-1.0.3 --features   # cache/ptbxl_100hz.npz + features.npz

    python train.py --model mlp                    # then cnn / bilstm / transformer (GPU recommended)
    python external.py --data-dir data/georgia     # cache/georgia_*.npz (needs PTB-XL cache: reuses its mu/sd)
    python external_eval.py                        # zero-shot Georgia scores for every trained model
    python efficiency.py                           # params / FLOPs / CPU latency
    python compare.py                              # results/results.md + figures (tables, CIs, error analysis)
    python run_all.py                              # or: train all 4 -> external eval -> efficiency -> compare in one go
    python -m pytest tests -q                      # synthetic-data unit + end-to-end smoke tests

Each `train.py` run writes `outputs/<model>/{metrics.json, val_probs.npy, test_probs.npy, best.pt}`
(+ `feat_norm.npz` for the MLP). Tune (`--lr`, `--model-kw '{"d_model":128}'`) on the **validation fold only**;
the test fold is scored once, after training, with thresholds fitted on val.

## Demo (after training)
    python make_demo_bundle.py        # packages sample ECGs + checkpoints + results into demo/ (~tens of MB)
    streamlit run app.py
Tab 1 runs all four models on one ECG (PTB-XL test or unseen-hospital Georgia) with per-model probabilities,
thresholds and CPU latency; tab 2 shows the benchmark graphs (per-class AUROC, in-distribution vs external,
accuracy-vs-cost, confusion matrices, ROC, CIs). `demo/` is self-contained, so it also deploys on Streamlit
Community Cloud / Hugging Face Spaces. Research demo, not a diagnostic tool.

## Design decisions worth defending
* **Multi-label**: sigmoid + BCE, not softmax. Records with no diagnostic superclass are dropped (as in the PTB-XL benchmark protocol).
* **No leakage**: per-lead z-score uses train-fold statistics only; folds are patient-disjoint; thresholds from val.
* **MLP uses real features**, not a flattened signal. Note its end-to-end latency is dominated by feature extraction (`efficiency.py` reports both).
* **External cohort = Georgia** (PhysioNet/CinC 2020), not CPSC-2018: CPSC-2018 has no hypertrophy and no MI-like class
  beyond ST changes, Georgia covers NORM/STTC/CD/HYP. Georgia has only ~7 MI records, so **MI is not scored externally**
  and external macro AUROC is compared with PTB-XL's over the *same* classes. SNOMED->superclass mapping is in `external.py`
  (verified against PTB-XL `scp_statements.csv`). The Georgia download is a seeded random subset because physionet.org is rate-limited.
* **External preprocessing** is identical to PTB-XL: 500->100 Hz polyphase resampling, first 10 s (zero-pad if short), 0.5 Hz high-pass, PTB-XL train mu/sd.
* **Uncertainty**: 95% CIs by patient-level bootstrap; model-vs-model comparisons by paired bootstrap, so "CNN beats BiLSTM" is only claimed when it is not noise.
* Accuracy is never reported: classes are imbalanced (a predict-NORM model looks good). Primary = macro AUROC; secondary = macro F1, per-class P/R/F1, and a multi-label confusion matrix.
