# MLP tuning log (M1, Saurabh Sharad)

Model: `models/mlp.py`, feed-forward net on the 154 engineered features from `data._record_features`.
Data: shared `cache/ptbxl_100hz.npz` (SHA-256 `675572bf7ad70b0af2b3c11d46f3e6802b301cbd9eeecb83fba50175b45e6b14`, verified),
`cache/features.npz` built locally with `python -c "import data; data.build_feature_cache()"` (68 s).
Environment: macOS, Apple M5 CPU, Python 3.12.15, torch 2.15.0, numpy 2.5.4, scikit-learn 1.9.1.

Protocol: shared defaults (seed 42, AdamW, BCE, batch 64, max 50 epochs, early stopping on val macro AUROC,
patience 8). Configs chosen on the **validation fold only**; the test fold was not looked at for any choice.

## Trials (6 configs, `--out outputs_trials/<name>`)

| Name | Changed from default | Params | Epochs run | Best epoch | Val macro AUROC | Val macro F1 |
|---|---|---|---|---|---|---|
| base | none (lr 1e-3, wd 1e-2, hidden 256-128, dropout 0.3) | 73,989 | 43 | 35 | 0.9007 | 0.7157 |
| **lr3e-4** | `--lr 3e-4` | 73,989 | 43 | 35 | **0.9023** | 0.7171 |
| lr3e-3 | `--lr 3e-3` | 73,989 | 23 | 15 | 0.8993 | 0.7069 |
| wide | `--model-kw '{"hidden":[512,256]}'` | 213,509 | 35 | 27 | 0.9021 | 0.7120 |
| deep | `--model-kw '{"hidden":[512,256,128]}'` | 246,021 | 26 | 18 | 0.9006 | 0.7121 |
| drop0.5_wd5e-2 | `--model-kw '{"dropout":0.5}' --weight-decay 5e-2` | 73,989 | 36 | 28 | 0.9002 | 0.7102 |

Selected: **lr3e-4** (highest val macro AUROC). All six are within 0.003 val AUROC of each other, so the MLP is
insensitive to these knobs; extra capacity (wide/deep) did not help, which suggests the features, not the
classifier, limit performance.

## Final run

    python train.py --model mlp --lr 3e-4

Written to `outputs/mlp/`. Re-running reproduced the trial exactly (identical val probabilities; CPU is deterministic).

| Split | Macro AUROC | Macro F1 |
|---|---|---|
| Val (fold 9) | 0.9023 | 0.7171 |
| Test (fold 10, scored once) | 0.8888 | 0.6890 |

Per-class test AUROC: NORM 0.915, MI 0.861, STTC 0.896, CD 0.883, HYP 0.889.
Val-fitted thresholds: NORM 0.48, MI 0.31, STTC 0.42, CD 0.40, HYP 0.45.

## Notes
- Feature quality: R-peak detection on lead II found fewer than 3 beats in 52 of 21,388 records (rhythm/HRV
  features set to 0 there), and detected heart rate spans 12.6-199.4 bpm (median 76.5), so some beats are missed.
- MI is the weakest class: infarction shows up as Q waves / ST shape in specific leads, which per-lead summary
  statistics and band powers capture only coarsely.
