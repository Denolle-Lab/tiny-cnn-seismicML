#!/usr/bin/env python3
"""
Draw the held-out inference subset: 10 windows per class from station-days
and catalog events no model was trained on (labeled_data/heldout/, written by
collect_continuous_windows.py and collect_ak_events.py).

Every window is cut to the 6000-sample model input (earthquakes start
P_LEAD_SEC before the P arrival) and stored as filtered counts; z-scoring is
left to inference, as in the training notebook.

Usage (from repo root):
  python scripts/make_heldout_subset.py

Output: datasets/heldout_inference/heldout_waveforms.npy  (n, 6000) float32, local only
            (git-ignored: the Raspberry Shake terms forbid redistributing waveforms)
        datasets/heldout_inference/heldout_metadata.csv   one row per window, committed
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.labels import LABEL_MAP

HELDOUT_DIR = REPO_ROOT / "notebooks" / "02_labeling" / "labeled_data" / "heldout"
OUT_DIR = REPO_ROOT / "datasets" / "heldout_inference"
PER_CLASS = 10
TARGET_LEN = 6000
SAMPLING_RATE = 100
P_LEAD_SEC = 10.0
SEED = 42

# Noise is shared across sites so no one sensor stands for the class.
NOISE_PER_SET = {"heldout_AK": 3, "heldout_AM_romig_traffic": 3,
                 "heldout_AM_aircraft": 2, "heldout_AK_train": 2}


def load_sets():
    """Concatenate every held-out file set; returns list of windows and one metadata frame."""
    windows, frames = [], []
    for wf in sorted(HELDOUT_DIR.glob("*_waveforms_*.npy")):
        prefix, stamp = wf.stem.split("_waveforms_")
        meta = pd.read_csv(HELDOUT_DIR / f"{prefix}_metadata_{stamp}.csv")
        X = np.load(wf, allow_pickle=True)
        meta["set"] = prefix.split("_20")[0]          # heldout_AK_train_2026-05-11 -> heldout_AK_train
        meta["source_file"] = wf.name
        meta["source_row"] = np.arange(len(meta))
        windows += [np.asarray(w, dtype=np.float32) for w in X]
        frames.append(meta)
    return windows, pd.concat(frames, ignore_index=True)


def crop(w, p_offset_sec):
    """Cut to TARGET_LEN; earthquake windows start P_LEAD_SEC before P."""
    if len(w) == TARGET_LEN:
        return w, 0.0
    start = int(round((p_offset_sec - P_LEAD_SEC) * SAMPLING_RATE))
    start = min(max(start, 0), len(w) - TARGET_LEN)
    return w[start:start + TARGET_LEN], start / SAMPLING_RATE


def main():
    windows, meta = load_sets()
    rng = np.random.default_rng(SEED)

    picks = []
    for label, name in LABEL_MAP.items():
        if name == "Noise":
            for set_name, n in NOISE_PER_SET.items():
                idx = meta.index[(meta.label == label) & (meta.set == set_name)]
                picks += list(rng.choice(idx, min(n, len(idx)), replace=False))
        else:
            idx = meta.index[meta.label == label]
            picks += list(rng.choice(idx, min(PER_CLASS, len(idx)), replace=False))

    X, crop_start = zip(*(crop(windows[i], meta.loc[i].get("p_offset_sec")) for i in picks))
    out = meta.loc[picks].reset_index(drop=True)
    out.insert(0, "subset_id", np.arange(len(out)))
    out["crop_start_sec"] = crop_start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "heldout_waveforms.npy", np.stack(X).astype(np.float32))
    out.to_csv(OUT_DIR / "heldout_metadata.csv", index=False)
    print(out.groupby(["label_name", "station"]).size().to_string())
    print(f"{len(out)} windows -> {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
