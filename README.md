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
| MLP | engineered features (RR/HRV, QRS width, per-lead stats, band powers) | `models/mlp.py` | Saurabh Sharad |
| 1D-CNN (ResNet) | raw 1000x12 | `models/cnn.py` | Riddhima Jain |
| BiLSTM | raw 1000x12 | `models/bilstm.py` | Ambica Mehra |
| Transformer encoder | raw 1000x12 (patch-embedded) | `models/transformer.py` | Nehal Verma |

## Results snapshot (PTB-XL test fold 10; full tables in `results/results.md`)
| Model | Macro AUROC [95% CI] | Macro F1 | Params | CPU latency (ms/ECG) |
|---|---|---|---|---|
| MLP (features) | 0.889 [0.878, 0.898] | 0.689 | 73,989 | 0.18 (50.1 incl. feature extraction) |
| 1D-CNN | 0.924 [0.916, 0.930] | 0.744 | 971,717 | 11.43 |
| BiLSTM | 0.924 [0.916, 0.930] | 0.738 | 613,605 | 29.21 |
| Transformer | 0.907 [0.897, 0.915] | 0.713 | 167,237 | 8.23 |
| Stacked fusion | 0.927 [0.920, 0.934] | 0.745 | - | - |

CNN and BiLSTM are statistically tied; the Transformer (default config, no tuning trials) is third; ordering holds on the
zero-shot Georgia cohort (all models lose 0.06-0.09 AUROC). Model errors are highly correlated (0.52-0.71), so fusion gains
only ~0.003 AUROC (`results/ensemble.md`). Each of MLP/CNN/BiLSTM had a 6-config tuning budget (logs in `tuning_logs/`,
`tuning/`, `results/bilstm/`); the Transformer was trained once with defaults (`tuning_logs/transformer_tuning_log.md`).

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
    python ensemble.py                             # soft/weighted voting, stacking, disagreement -> results/ensemble.md
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

## Reproducing the results without retraining
Final per-model predictions are committed: `results/final/{mlp,cnn,transformer}/` and `results/bilstm/final/`
(`metrics.json`, `val_probs.npy`, `test_probs.npy`). Copy them to `outputs/<model>/` (BiLSTM from `results/bilstm/final/`),
build the signal cache (`data.py`), then run `compare.py` and `ensemble.py`. Checkpoints (`best.pt`) are not committed except for the BiLSTM.

## Repository map
* `data.py`, `train.py`, `evaluate.py`: shared preprocessing, training loop, metrics (do not edit per model).
* `models/`: the four architectures. `external*.py`, `efficiency.py`, `compare.py`, `ensemble.py`: analysis.
* `results/`: generated tables/figures, plus final predictions. `tuning_logs/`, `tuning/`: per-model tuning notes.
* `report/`: project report (draft). `HANDOFF.md`, `KAGGLE.md`: training instructions for the team.
