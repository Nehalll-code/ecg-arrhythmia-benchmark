# Train CNN / BiLSTM / Transformer on Kaggle (GPU)

The MLP, external evaluation, efficiency and comparison run on your own machine afterwards.

## One-time setup
1. **Upload the cache as a Kaggle Dataset** (kaggle.com > Datasets > New Dataset): upload
   `cache/ptbxl_100hz.npz` (957 MB). Name it e.g. `ptbxl-100hz-cache`.
   SHA-256 must be `675572bf7ad70b0af2b3c11d46f3e6802b301cbd9eeecb83fba50175b45e6b14`.
2. **Upload the code as a second Dataset** (zip of the project folder WITHOUT `data/`, `cache/`,
   `outputs/`, `logs/`), e.g. `ecg-code`. Or `git clone` it in the notebook if the repo is pushed.
3. New Notebook > right side **Settings**: Accelerator = **GPU T4 x2 or P100**, Persistence off is fine.
   **Add Input**: both datasets.

## Notebook cells
```python
# 1) code + check the data
!cp -r /kaggle/input/ecg-code/* /kaggle/working/ && cd /kaggle/working && ls
!sha256sum /kaggle/input/ptbxl-100hz-cache/ptbxl_100hz.npz
!pip install -q wfdb      # only needed to import data.py; torch/sklearn/scipy are preinstalled
```
```python
# 2) sanity: GPU visible, 1 quick epoch of the transformer
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
!cd /kaggle/working && python train.py --model transformer --epochs 1 --data /kaggle/input/ptbxl-100hz-cache/ptbxl_100hz.npz --out outputs_smoke
```
```python
# 3) the real runs (resumable: re-run the cell, finished models are skipped)
import os, subprocess
DATA = "/kaggle/input/ptbxl-100hz-cache/ptbxl_100hz.npz"
for m in ["cnn", "bilstm", "transformer"]:
    if os.path.exists(f"/kaggle/working/outputs/{m}/metrics.json"):
        continue
    subprocess.run(["python", "train.py", "--model", m, "--data", DATA], cwd="/kaggle/working", check=True)
```
```python
# 4) package the results, then download /kaggle/working/outputs.zip from the Output tab
!cd /kaggle/working && zip -r outputs.zip outputs -x "*.pyc"
```

## Bring results back
Unzip `outputs.zip` into this project's `outputs/` (you get `outputs/{cnn,bilstm,transformer}/`).
Then locally: `python run_all.py` (trains the MLP, skips the three finished models, then does
external eval + efficiency + comparison).

## Notes
- Same seed (42) and defaults as the shared protocol; do not change them for one model only.
- Tuning: only on the validation fold, same small budget for each model (see HANDOFF.md).
- Kaggle sessions cap at ~9-12 h and ~30 GPU-h/week; my rough guess is tens of minutes per model on a T4/P100 (not measured; step 2 of the notebook gives you a real per-epoch time to extrapolate from).
  If a session dies, `--epochs`/early stopping state is not resumable mid-model, so rerun that model.
- Everything stays private: uploaded datasets are private unless you make them public.
