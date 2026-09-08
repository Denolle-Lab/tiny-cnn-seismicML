"""
Offline end-to-end test of the multi-class pipeline on synthetic traces.

Runs without network access: synthesizes four signal classes, cuts
windows with src.data.windows, merges runs, trains for a few epochs with
scripts/train_multiclass.py, packages with scripts/package_model.py and
checks the CLUE metadata and TF.js weight keys.

Run:  python tests/test_pipeline.py        (or pytest tests/)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import (preprocess_trace, cut_event_and_noise, sta_lta_onset,  # noqa: E402
                              save_dataset, load_dataset, merge_datasets,
                              event_level_split, expand_crops)

FS = 100.0
CLASSES = ["Noise", "Earthquake", "Blast", "Ice quake"]


def synth_trace(kind, rng, fs=FS, dur=240.0, onset=120.0):
    """Coloured noise plus a class-specific transient at ``onset`` seconds."""
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = rng.normal(0, 1.0, n)
    x = np.convolve(x, np.ones(5) / 5, mode="same")  # slightly reddened noise
    i0 = int(onset * fs)
    tt = t[i0:] - onset
    if kind == "Earthquake":
        env = 40 * tt * np.exp(-tt / 15.0)
        x[i0:] += env * np.sin(2 * np.pi * 5.0 * tt)
    elif kind == "Blast":
        env = 60 * np.exp(-tt / 2.0)
        x[i0:] += env * np.sin(2 * np.pi * 12.0 * tt)
    elif kind == "Ice quake":
        env = 25 * (1 - np.exp(-tt / 3.0)) * np.exp(-tt / 25.0)
        x[i0:] += env * np.sin(2 * np.pi * 3.0 * tt)
    return x


def build_run(out_dir, kind, n_events, n_stations, rng, class_index):
    windows, labels, rows = [], [], []
    for e in range(n_events):
        eid = f"{kind[:2].lower()}{e:04d}"
        for s in range(n_stations):
            raw = synth_trace(kind, rng)
            data = preprocess_trace(raw, FS, FS, 2.0, 20.0)
            guess = int(120.0 * FS) + int(rng.integers(-150, 150))
            onset = sta_lta_onset(data, FS, guess, search_sec=3.0) or guess
            ev, nz = cut_event_and_noise(data, FS, onset)
            assert ev is not None and nz is not None
            base = dict(event_id=eid, network="AK" if s % 2 == 0 else "AM", station=f"S{s}",
                        channel="BHZ", magnitude=2.0, distance_km=50.0, onset_source="test")
            windows.append(ev); labels.append(class_index)
            rows.append(dict(base, label_name=kind, window_type="event", onset_sample=3000, window_len=len(ev)))
            windows.append(nz); labels.append(0)
            rows.append(dict(base, label_name="Noise", window_type="noise", onset_sample=-1, window_len=len(nz)))
    save_dataset(out_dir, windows, labels, pd.DataFrame(rows), CLASSES,
                 {"sampling_rate": 100, "train_len_sec": 60, "lowcut_hz": 2.0, "highcut_hz": 20.0})
    return out_dir


def test_windows_basic():
    rng = np.random.default_rng(0)
    raw = synth_trace("Blast", rng)
    data = preprocess_trace(raw, FS, FS)
    assert len(data) == len(raw)
    onset = sta_lta_onset(data, FS, int(120 * FS) + 100, search_sec=3.0)
    assert abs(onset - 120 * FS) < 0.5 * FS, f"STA/LTA onset off by {onset - 120 * FS} samples"
    ev, nz = cut_event_and_noise(data, FS, onset)
    assert len(ev) == 12000 and len(nz) == 6000
    assert abs(ev.mean()) < 1e-3 and abs(ev.std() - 1) < 1e-3
    # resampling path
    d50 = preprocess_trace(raw[::2], 50.0, 100.0)
    assert abs(len(d50) - len(raw)) <= 2


def test_event_level_split_has_no_leak():
    rng = np.random.default_rng(1)
    ids = np.repeat([f"e{i}" for i in range(40)], 3)
    rng.shuffle(ids)
    m = event_level_split(ids, (0.7, 0.15, 0.15), seed=3)
    assert m["train"].sum() + m["val"].sum() + m["test"].sum() == len(ids)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(ids[m[a]]) & set(ids[m[b]])), f"event overlap between {a} and {b}"
    assert m["val"].any() and m["test"].any()


def test_expand_crops():
    X = [np.random.randn(12000).astype(np.float32), np.random.randn(6000).astype(np.float32)]
    Xc, yc, src = expand_crops(X, [1, 0], [3000, -1], 6000, n_onset_crops=2, n_free_crops=1)
    assert Xc.shape == (4, 6000) and list(yc) == [1, 1, 1, 0] and list(src) == [0, 0, 0, 1]


def test_end_to_end(tmp_root=None):
    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="tinycnn_"))
    rng = np.random.default_rng(42)
    runs = []
    for kind, n_ev in (("Earthquake", 14), ("Blast", 12), ("Ice quake", 12)):
        runs.append(build_run(tmp_root / f"run_{kind.replace(' ', '_')}", kind, n_ev, 2, rng, CLASSES.index(kind)))

    # exclusion file path in merge script
    (runs[0] / "exclude.txt").write_text("0\n1\n")

    windows, labels, meta, info = merge_datasets(runs, CLASSES, {"Earthquake": 20})
    assert set(labels) == {0, 1, 2, 3}
    assert sum(1 for l in labels if l == 1) <= 20
    ds = tmp_root / "dataset"
    save_dataset(ds, windows, labels, meta, CLASSES, info["window"])
    X, y, meta2, info2 = load_dataset(ds)
    assert len(X) == len(y) == len(meta2) and info2["classes"] == CLASSES

    out = tmp_root / "models"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_multiclass.py"), "--dataset", str(ds),
           "--epochs", "6", "--patience", "6", "--batch-size", "16", "--out", str(out), "--tag", "synthetic"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    ckpts = sorted(out.glob("seismic_cnn_standard_*.pth"))
    assert ckpts, r.stdout
    metrics = json.load(open(sorted(out.glob("metrics_standard_*.json"))[-1]))
    assert metrics["classes"] == CLASSES and len(metrics["confusion"]) == 4
    assert "AK" in metrics["by_network"] and "AM" in metrics["by_network"]
    print("  synthetic test accuracy (standard):", round(metrics["test_accuracy"], 3),
          "recall:", [round(x, 2) for x in metrics["test_recall"]])

    pk = [sys.executable, str(REPO_ROOT / "scripts" / "package_model.py"), "--checkpoint", str(ckpts[-1]),
          "--model-id", "standard-test", "--out-root", str(tmp_root / "pkg")]
    r = subprocess.run(pk, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    md = json.load(open(tmp_root / "pkg" / "standard-test" / "metadata.json"))
    assert md["class_names"] == CLASSES and md["architecture"] == "standard"
    assert md["sampling_rate"] == 100 and md["window_duration"] == 60 and md["instrument_types"] == ["H"]
    w = json.load(open(tmp_root / "pkg" / "standard-test" / "weights.json"))
    assert w["num_classes"] == 4
    for k in ("conv1/kernel", "bn4/moving_variance", "fc2/kernel", "fc2/bias"):
        assert k in w, k
    assert len(w["fc2/bias"]) == 4 and len(w["fc2/kernel"]) == 64

    # the compact exporter path too
    ck = sorted(out.glob("seismic_cnn_compact_*.pth"))
    assert ck
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "package_model.py"), "--checkpoint",
                        str(ck[-1]), "--model-id", "compact-test", "--out-root", str(tmp_root / "pkg"),
                        "--instrument-types", "H,L"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    md = json.load(open(tmp_root / "pkg" / "compact-test" / "metadata.json"))
    assert md["instrument_types"] == ["H", "L"] and md["architecture"] == "compact"
    print("  end-to-end OK in", tmp_root)


if __name__ == "__main__":
    for fn in (test_windows_basic, test_event_level_split_has_no_leak, test_expand_crops, test_end_to_end):
        print(f"{fn.__name__} ...")
        fn()
        print("  ok")
    print("all tests passed")
