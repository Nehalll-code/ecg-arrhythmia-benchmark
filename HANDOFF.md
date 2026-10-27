# Training handoff (one model each)

Everyone trains on the **identical cache**; if the file differs the comparison is invalid.

## 1. Get the code and data
    git clone <repo> && cd <repo>
    pip install -r requirements.txt        # install the CUDA build of torch first if you have a GPU
Download the shared cache files into `cache/` (link from the group lead):

| File | Needed by | SHA-256 |
|---|---|---|
| `cache/ptbxl_100hz.npz` (957 MB) | everyone (M1 also needs it to build features) | `675572bf7ad70b0af2b3c11d46f3e6802b301cbd9eeecb83fba50175b45e6b14` |
| `cache/features.npz` | M1 MLP | build it yourself from the signal cache (below); needs ~3 GB free RAM, ~20 min |

M1 (MLP) builds the features after placing `ptbxl_100hz.npz` in `cache/`:

    python -c "import data; data.build_feature_cache()"

Verify: `sha256sum cache/ptbxl_100hz.npz` (Windows: `certutil -hashfile cache\ptbxl_100hz.npz SHA256`).
Do NOT rebuild the signal cache from raw PTB-XL and do NOT edit `data.py`, `train.py`, `evaluate.py`.

## 2. Train (GPU strongly recommended for M2-M4)
    python train.py --model cnn            # or bilstm / transformer / mlp
Defaults (seed 42, AdamW, BCE, early stopping on val macro AUROC, patience 8) are the shared protocol.

**Tuning rules:** tune only on the **validation** fold (the script already reports it); the test fold is
never used for choices. Allowed knobs: `--lr`, `--weight-decay`, `--batch-size`, `--model-kw '{"d_model":128}'`.
Give each trial its own folder with `--out outputs_trials/<name>`, keep a note of every trial and its
val AUROC (this is your tuning log for the report), then re-run the best config once into the default `outputs/`.
Budget: at most ~6 configs per model so nobody gets a bigger tuning budget than the others.

## 3. Send back (zip the whole folder `outputs/<model>/`)
- `metrics.json`, `val_probs.npy`, `test_probs.npy`
- `best.pt` (needed for the external-cohort and efficiency runs)
- `feat_norm.npz` (MLP only)
- your tuning log (every config tried + val AUROC)

Commit your own model file / tuning notes under your own name so the git history shows who did what.
Be ready to explain your model in the viva: what it assumes about where the diagnostic information is,
why it should or shouldn't beat the others, and what the per-class results show.
