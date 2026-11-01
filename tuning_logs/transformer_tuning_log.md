# Transformer encoder (M4, Nehal Verma): training log

No hyperparameter search was run. Only the default shared-protocol configuration was trained.

| Config | Command | Best val macro AUROC | Test macro AUROC |
|---|---|---|---|
| default (lr 1e-3, wd 1e-2, bs 64, seed 42; d_model 64, 4 heads, 3 layers, patch 5) | `python train.py --model transformer` | 0.9165 (epoch 11) | 0.9066 |

Early stopping (patience 8) ended training at epoch 19 on a Kaggle T4 (about 8 s/epoch).
Training loss kept falling after epoch 11 while val AUROC plateaued, so the model overfits mildly.
Untried and left as future work: lower lr, higher dropout, a larger d_model.
