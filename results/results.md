# Results

## 1. PTB-XL test (fold 10), in-distribution

| Model | Macro AUROC [95% CI] | Macro F1 | AUROC NORM | AUROC MI | AUROC STTC | AUROC CD | AUROC HYP |
|---|---|---|---|---|---|---|---|
| MLP | 0.889 [0.878, 0.898] | 0.689 | 0.915 | 0.861 | 0.896 | 0.883 | 0.889 |
| 1D-CNN | 0.924 [0.916, 0.930] | 0.744 | 0.949 | 0.919 | 0.935 | 0.919 | 0.898 |
| BiLSTM | 0.924 [0.916, 0.930] | 0.738 | 0.945 | 0.918 | 0.940 | 0.920 | 0.893 |
| Transformer | 0.907 [0.897, 0.915] | 0.713 | 0.934 | 0.895 | 0.927 | 0.889 | 0.888 |
| Ensemble (mean) | 0.925 [0.918, 0.932] | 0.749 | 0.947 | 0.915 | 0.940 | 0.920 | 0.904 |

P(row > column), paired patient-level bootstrap of macro AUROC (~0.5 = indistinguishable; >0.95 or <0.05 = clear difference):

| | MLP | 1D-CNN | BiLSTM | Transformer | Ensemble (mean) |
|---|---|---|---|---|---|
| MLP | - | 0.00 | 0.00 | 0.00 | 0.00 |
| 1D-CNN | 1.00 | - | 0.54 | 1.00 | 0.13 |
| BiLSTM | 1.00 | 0.46 | - | 1.00 | 0.09 |
| Transformer | 1.00 | 0.00 | 0.00 | - | 0.00 |
| Ensemble (mean) | 1.00 | 0.87 | 0.91 | 1.00 | - |

## 2. Per-class precision / recall / F1 (thresholds tuned on val fold)

| Model | Class | n+ | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MLP | NORM | 963 | 0.776 | 0.871 | 0.821 |
| MLP | MI | 550 | 0.584 | 0.729 | 0.648 |
| MLP | STTC | 521 | 0.657 | 0.712 | 0.683 |
| MLP | CD | 496 | 0.770 | 0.655 | 0.708 |
| MLP | HYP | 262 | 0.675 | 0.515 | 0.584 |
| 1D-CNN | NORM | 963 | 0.810 | 0.933 | 0.867 |
| 1D-CNN | MI | 550 | 0.710 | 0.749 | 0.729 |
| 1D-CNN | STTC | 521 | 0.732 | 0.783 | 0.757 |
| 1D-CNN | CD | 496 | 0.779 | 0.732 | 0.755 |
| 1D-CNN | HYP | 262 | 0.585 | 0.641 | 0.612 |
| BiLSTM | NORM | 963 | 0.773 | 0.958 | 0.856 |
| BiLSTM | MI | 550 | 0.753 | 0.705 | 0.729 |
| BiLSTM | STTC | 521 | 0.720 | 0.831 | 0.772 |
| BiLSTM | CD | 496 | 0.822 | 0.710 | 0.762 |
| BiLSTM | HYP | 262 | 0.494 | 0.679 | 0.572 |
| Transformer | NORM | 963 | 0.817 | 0.879 | 0.846 |
| Transformer | MI | 550 | 0.652 | 0.725 | 0.687 |
| Transformer | STTC | 521 | 0.736 | 0.760 | 0.748 |
| Transformer | CD | 496 | 0.818 | 0.617 | 0.703 |
| Transformer | HYP | 262 | 0.598 | 0.561 | 0.579 |
| Ensemble (mean) | NORM | 963 | 0.810 | 0.925 | 0.864 |
| Ensemble (mean) | MI | 550 | 0.756 | 0.704 | 0.729 |
| Ensemble (mean) | STTC | 521 | 0.751 | 0.818 | 0.783 |
| Ensemble (mean) | CD | 496 | 0.786 | 0.724 | 0.753 |
| Ensemble (mean) | HYP | 262 | 0.607 | 0.626 | 0.617 |

## 3. External validation: Georgia (n=2059), zero-shot, no re-tuning

Scored classes (>= 30 positives): NORM, STTC, CD, HYP. Not scored: MI: n+=1.

Both macro AUROCs average over the SAME classes, so the drop is like-for-like.

| Model | PTB-XL test | Georgia | Delta | Georgia rank | PTB-XL rank | NORM | STTC | CD | HYP |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.896 [0.884, 0.906] | 0.813 [0.802, 0.825] | -0.083 | 5 | 5 | 0.868 | 0.768 | 0.827 | 0.787 |
| 1D-CNN | 0.925 [0.916, 0.932] | 0.862 [0.852, 0.872] | -0.063 | 1 | 2 | 0.899 | 0.835 | 0.905 | 0.807 |
| BiLSTM | 0.925 [0.915, 0.932] | 0.846 [0.836, 0.856] | -0.079 | 3 | 3 | 0.892 | 0.835 | 0.894 | 0.763 |
| Transformer | 0.910 [0.899, 0.919] | 0.824 [0.813, 0.835] | -0.086 | 4 | 4 | 0.887 | 0.815 | 0.838 | 0.756 |
| Ensemble (mean) | 0.928 [0.919, 0.935] | 0.853 [0.843, 0.863] | -0.075 | 2 | 1 | 0.899 | 0.832 | 0.886 | 0.795 |

P(row > column) on Georgia (paired record-level bootstrap):

| | MLP | 1D-CNN | BiLSTM | Transformer | Ensemble (mean) |
|---|---|---|---|---|---|
| MLP | - | 0.00 | 0.00 | 0.00 | 0.00 |
| 1D-CNN | 1.00 | - | 1.00 | 1.00 | 1.00 |
| BiLSTM | 1.00 | 0.00 | - | 1.00 | 0.00 |
| Transformer | 1.00 | 0.00 | 0.00 | - | 0.00 |
| Ensemble (mean) | 1.00 | 0.00 | 1.00 | 1.00 | - |

**Winner in-distribution (same classes): 1D-CNN; on Georgia: 1D-CNN.** The winner persists under distribution shift.

## 4. Efficiency (per ECG; CPU, 1 thread, batch 1)

| Model | Params | MFLOPs | Latency model-only (ms) | Latency end-to-end (ms) | Macro AUROC |
|---|---|---|---|---|---|
| MLP | 73,989 | 0.1 | 0.18 | 50.10 | 0.889 |
| 1D-CNN | 971,717 | 178.5 | 11.43 | 11.43 | 0.924 |
| BiLSTM | 613,605 | 304.8 | 29.21 | 29.21 | 0.924 |
| Transformer | 167,237 | 60.8 | 8.23 | 8.23 | 0.907 |

MLP end-to-end includes hand-crafted feature extraction (49.9 ms/ECG).

## 5. Error analysis (PTB-XL test)

Rows = true class; columns = what the model predicted for those records (multi-label, so a row can sum to >1). Diagonal = recall; off-diagonal = confusion with another class; 'none' = nothing predicted.

**MLP**

| true \ predicted | NORM | MI | STTC | CD | HYP | none |
|---|---|---|---|---|---|---|
| NORM | 0.87 | 0.08 | 0.08 | 0.04 | 0.01 | 0.01 |
| MI | 0.18 | 0.73 | 0.33 | 0.33 | 0.14 | 0.01 |
| STTC | 0.15 | 0.45 | 0.71 | 0.18 | 0.25 | 0.01 |
| CD | 0.23 | 0.51 | 0.22 | 0.66 | 0.11 | 0.01 |
| HYP | 0.16 | 0.49 | 0.62 | 0.25 | 0.52 | 0.01 |

TP/FP/FN/TN: NORM: 839/242/124/953; MI: 401/286/149/1322; STTC: 371/194/150/1443; CD: 325/97/171/1565; HYP: 135/65/127/1831

**1D-CNN**

| true \ predicted | NORM | MI | STTC | CD | HYP | none |
|---|---|---|---|---|---|---|
| NORM | 0.93 | 0.04 | 0.04 | 0.06 | 0.02 | 0.01 |
| MI | 0.15 | 0.75 | 0.35 | 0.32 | 0.20 | 0.01 |
| STTC | 0.08 | 0.37 | 0.78 | 0.20 | 0.36 | 0.02 |
| CD | 0.21 | 0.43 | 0.26 | 0.73 | 0.13 | 0.00 |
| HYP | 0.16 | 0.36 | 0.57 | 0.27 | 0.64 | 0.01 |

TP/FP/FN/TN: NORM: 898/210/65/985; MI: 412/168/138/1440; STTC: 408/149/113/1488; CD: 363/103/133/1559; HYP: 168/119/94/1777

**BiLSTM**

| true \ predicted | NORM | MI | STTC | CD | HYP | none |
|---|---|---|---|---|---|---|
| NORM | 0.96 | 0.03 | 0.04 | 0.05 | 0.02 | 0.00 |
| MI | 0.19 | 0.71 | 0.37 | 0.29 | 0.22 | 0.01 |
| STTC | 0.12 | 0.33 | 0.83 | 0.18 | 0.38 | 0.00 |
| CD | 0.25 | 0.39 | 0.27 | 0.71 | 0.29 | 0.00 |
| HYP | 0.23 | 0.35 | 0.63 | 0.26 | 0.68 | 0.00 |

TP/FP/FN/TN: NORM: 923/271/40/924; MI: 388/127/162/1481; STTC: 433/168/88/1469; CD: 352/76/144/1586; HYP: 178/182/84/1714

**Transformer**

| true \ predicted | NORM | MI | STTC | CD | HYP | none |
|---|---|---|---|---|---|---|
| NORM | 0.88 | 0.06 | 0.04 | 0.03 | 0.01 | 0.02 |
| MI | 0.14 | 0.73 | 0.33 | 0.28 | 0.15 | 0.03 |
| STTC | 0.09 | 0.39 | 0.76 | 0.15 | 0.31 | 0.02 |
| CD | 0.18 | 0.46 | 0.22 | 0.62 | 0.14 | 0.02 |
| HYP | 0.15 | 0.41 | 0.60 | 0.22 | 0.56 | 0.01 |

TP/FP/FN/TN: NORM: 846/190/117/1005; MI: 399/213/151/1395; STTC: 396/142/125/1495; CD: 306/68/190/1594; HYP: 147/99/115/1797

**Ensemble (mean)**

| true \ predicted | NORM | MI | STTC | CD | HYP | none |
|---|---|---|---|---|---|---|
| NORM | 0.93 | 0.03 | 0.04 | 0.05 | 0.02 | 0.01 |
| MI | 0.16 | 0.70 | 0.35 | 0.34 | 0.18 | 0.03 |
| STTC | 0.08 | 0.32 | 0.82 | 0.19 | 0.34 | 0.02 |
| CD | 0.21 | 0.39 | 0.24 | 0.72 | 0.15 | 0.01 |
| HYP | 0.17 | 0.34 | 0.60 | 0.29 | 0.63 | 0.01 |

TP/FP/FN/TN: NORM: 891/209/72/986; MI: 387/125/163/1483; STTC: 426/141/95/1496; CD: 359/98/137/1564; HYP: 164/106/98/1790
