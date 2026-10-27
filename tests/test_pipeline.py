"""Unit + end-to-end smoke tests on synthetic data (no downloads needed).

    python -m pytest tests -q
The smoke test runs train -> efficiency -> external_eval -> compare in a temp working dir
(the scripts use cwd-relative cache/ and outputs/ paths).
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import CLASSES, _record_features  # noqa: E402
from evaluate import (best_thresholds, bootstrap_macro_auroc, confusion_matrix_multilabel,  # noqa: E402
                      evaluate)
from external import fix_length, map_labels, to_100hz  # noqa: E402
from models import build_model  # noqa: E402


def synth(n, rng, fold=None):
    """Signals whose lead-wise sine frequency encodes the labels, so models can learn them."""
    y = (rng.random((n, 5)) < 0.3).astype(np.float32)
    y[y.sum(1) == 0, 0] = 1
    t = np.arange(1000) / 100.0
    X = rng.standard_normal((n, 1000, 12)).astype(np.float32) * 0.5
    for c in range(5):
        X += (y[:, c][:, None, None] * np.sin(2 * np.pi * (1 + c) * t)[None, :, None]).astype(np.float32)
    return X, y


# ------------------------------------------------------------------ units
@pytest.mark.parametrize("name", ["cnn", "bilstm", "transformer"])
def test_signal_models_shape(name):
    out = build_model(name)(torch.randn(3, 1000, 12))
    assert out.shape == (3, 5)


def test_mlp_shape_and_feature_dim():
    f = _record_features(np.random.default_rng(0).standard_normal((1000, 12)).astype(np.float32))
    assert np.isfinite(f).all()
    assert build_model("mlp", in_dim=len(f)).eval()(torch.randn(2, len(f))).shape == (2, 5)


def test_label_mapping():
    assert map_labels(["426783006"]).tolist() == [1, 0, 0, 0, 0]               # NSR only -> NORM
    assert map_labels(["426783006", "164873001"]).tolist() == [0, 0, 0, 0, 1]  # NSR + LVH -> HYP only
    assert map_labels(["164909002", "429622005"]).tolist() == [0, 0, 1, 1, 0]  # LBBB + STD -> STTC, CD
    assert map_labels(["164889003"]).sum() == 0                                # AF alone: no superclass
    assert map_labels(["164865005"])[CLASSES.index("MI")] == 1


def test_resample_and_length():
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    out = to_100hz(np.random.randn(5000, 12), 500, names)
    assert out.shape == (1000, 12)
    assert to_100hz(np.random.randn(5000, 11), 500, names[:11]) is None
    assert fix_length(np.ones((700, 12), dtype=np.float32)).shape == (1000, 12)
    assert fix_length(np.ones((1500, 12), dtype=np.float32)).shape == (1000, 12)


def test_evaluate_perfect_and_missing_class():
    y = np.tile(np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]],
                         dtype=np.float32), (3, 1))
    assert evaluate(y, y.copy())["macro_auroc"] == 1.0
    y2 = y.copy()
    y2[:, 1] = 0  # class absent -> NaN AUROC skipped, not a crash
    r = evaluate(y2, np.random.rand(12, 5))
    assert r["per_class"][CLASSES[1]]["auroc"] is None and np.isfinite(r["macro_auroc"])


def test_thresholds_and_confusion():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 5)) < 0.3).astype(np.float32)
    p = np.clip(y * 0.6 + rng.random((500, 5)) * 0.4, 0, 1)
    th = best_thresholds(y, p)
    cm = confusion_matrix_multilabel(y, p, th)
    assert cm.shape == (5, 6) and (np.diag(cm[:, :5]) > 0.8).all()


def test_bootstrap_is_paired_and_patient_level():
    rng = np.random.default_rng(1)
    y = (rng.random((400, 5)) < 0.4).astype(np.float32)
    good, bad = np.clip(y + rng.normal(0, .3, y.shape), 0, 1), rng.random(y.shape)
    groups = np.repeat(np.arange(200), 2)
    b = bootstrap_macro_auroc(y, {"good": good, "bad": bad}, groups=groups, n=50)
    assert len(b["good"]) == len(b["bad"]) and (b["good"] > b["bad"]).all()


# ------------------------------------------------------------- end to end
def run(args, cwd):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{args}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
    return r.stdout


def test_end_to_end_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n = 330
    X, y = synth(n, rng)
    fold = np.repeat(np.arange(1, 11), n // 10)
    cache = tmp_path / "cache"
    cache.mkdir()
    pid = np.arange(n)  # one record per patient
    np.savez_compressed(cache / "ptbxl_100hz.npz", X=X, y=y, fold=fold, ecg_id=np.arange(n),
                        patient_id=pid, mu=np.zeros((1, 1, 12), np.float32),
                        sd=np.ones((1, 1, 12), np.float32))
    F = np.stack([_record_features(x) for x in X])
    np.savez_compressed(cache / "features.npz", F=F, y=y, fold=fold)
    Xe, ye = synth(120, rng)
    np.savez_compressed(cache / "georgia_100hz.npz", X=Xe, y=ye, ids=np.arange(120))
    np.savez_compressed(cache / "georgia_features.npz", F=np.stack([_record_features(x) for x in Xe]), y=ye)

    for m in ["mlp", "cnn", "bilstm", "transformer"]:
        run([os.path.join(ROOT, "train.py"), "--model", m, "--epochs", "2", "--batch-size", "32"], tmp_path)
        assert (tmp_path / "outputs" / m / "metrics.json").exists()
    run([os.path.join(ROOT, "efficiency.py"), "--runs", "5"], tmp_path)
    run([os.path.join(ROOT, "external_eval.py")], tmp_path)
    out = run([os.path.join(ROOT, "compare.py"), "--n-boot", "30"], tmp_path)
    assert "External validation" in out and "Efficiency" in out and "Error analysis" in out
    assert (tmp_path / "results" / "results.md").exists()
