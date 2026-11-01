# Fusion and complementarity (PTB-XL fold-10 test)

All fusion parameters are fitted on the validation fold. Best single model: **1D-CNN**. 'P(beats best single)' is the paired patient-level bootstrap probability that the fused macro AUROC exceeds the best single model's (about 0.5 = indistinguishable).

| System | Macro AUROC [95% CI] | Macro F1 | P(beats best single) |
|---|---|---|---|
| MLP | 0.889 [0.878, 0.898] | 0.689 | - |
| 1D-CNN | 0.924 [0.916, 0.930] | 0.744 | - |
| BiLSTM | 0.924 [0.916, 0.930] | 0.738 | - |
| Transformer | 0.907 [0.897, 0.915] | 0.713 | - |
| Soft vote (mean) | 0.925 [0.918, 0.932] | 0.749 | 0.87 |
| Weighted vote | 0.926 [0.918, 0.932] | 0.748 | 0.90 |
| Stacked LR | 0.927 [0.920, 0.934] | 0.745 | 1.00 |

Weighted-vote weights (from val macro AUROC - 0.5): {'mlp': 0.2396, 'cnn': 0.256, 'bilstm': 0.2564, 'transformer': 0.248}

## Per-class AUROC

| System | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| MLP | 0.915 | 0.861 | 0.896 | 0.883 | 0.889 |
| 1D-CNN | 0.949 | 0.919 | 0.935 | 0.919 | 0.898 |
| BiLSTM | 0.945 | 0.918 | 0.940 | 0.920 | 0.893 |
| Transformer | 0.934 | 0.895 | 0.927 | 0.889 | 0.888 |
| Soft vote (mean) | 0.947 | 0.915 | 0.940 | 0.920 | 0.904 |
| Weighted vote | 0.947 | 0.916 | 0.940 | 0.920 | 0.904 |
| Stacked LR | 0.949 | 0.923 | 0.942 | 0.923 | 0.900 |

## Pairwise prediction disagreement

Fraction of (record, class) decisions on which two models differ (binary predictions, val-tuned thresholds).

| | MLP | 1D-CNN | BiLSTM | Transformer |
|---|---|---|---|---|
| MLP | 0.000 | 0.109 | 0.115 | 0.111 |
| 1D-CNN | 0.109 | 0.000 | 0.062 | 0.085 |
| BiLSTM | 0.115 | 0.062 | 0.000 | 0.084 |
| Transformer | 0.111 | 0.085 | 0.084 | 0.000 |

## Error correlation

Pearson correlation of the per-decision error indicators (1 = identical mistakes, 0 = independent).

| | MLP | 1D-CNN | BiLSTM | Transformer |
|---|---|---|---|---|
| MLP | 1.00 | 0.54 | 0.52 | 0.54 |
| 1D-CNN | 0.54 | 1.00 | 0.71 | 0.61 |
| BiLSTM | 0.52 | 0.71 | 1.00 | 0.62 |
| Transformer | 0.54 | 0.61 | 0.62 | 1.00 |

## Oracle coverage

Of 10,790 (record, class) decisions: all four models wrong on 5.8%; at least one model right on 94.2%. Best single-model decision accuracy is 88.0% (1D-CNN). Hard majority vote (>=2 of 4) decision accuracy: 87.8%.
