# BiLSTM Hyperparameter Tuning Log

Dataset: PTB-XL 100 Hz shared cache
Seed: 42
Selection metric: Validation macro AUROC

| Trial | LR | Weight decay | Batch size | Model kwargs | Best validation AUROC | Epochs |
|---|---:|---:|---:|---|---:|---:|
| t0_default | 0.001 | 0.01 | 64 | `{}` | 0.9295 | 21 |
| t1_lr5e-4 | 0.0005 | 0.01 | 64 | `{}` | 0.9286 | 30 |
| t2_lr2e-3 | 0.002 | 0.01 | 64 | `{}` | 0.9312 | 22 |
| t3_bs128 | 0.001 | 0.01 | 128 | `{}` | 0.9279 | 29 |
| t4_wd1e-3 | 0.001 | 0.001 | 64 | `{}` | 0.9283 | 27 |
| t5_hidden96 | 0.001 | 0.01 | 64 | `{'hidden': 96}` | 0.9283 | 23 |
