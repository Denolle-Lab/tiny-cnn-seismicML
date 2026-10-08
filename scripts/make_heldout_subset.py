#!/usr/bin/env python3
"""
Draw the held-out inference subset: PER_CLASS windows per class from
station-days and catalog events no model was trained on (labeled_data/heldout/,
written by collect_continuous_windows.py and collect_ak_events.py).

Every candidate is checked against the committed split manifests
(datasets/splits/split_*.csv): a window whose group (catalog event, or
station + local day, as in the training notebook) or whose exact station and
start time appears in any train, val or test split is rejected. Positives are
spread evenly across stations; Noise is spread across the four sites.

Train is the exception: K222 passenger service ended 2026-09-14, so only a
couple of passages are unseen. The class is filled from the grouped test
split of the only models with a Train class (compact-v4 / standard-v3,
TRAIN_TEST_MANIFEST): K222 days those two never trained or validated on.
Other runs may have trained on Noise from those days; Train is out of scope
for them. heldout_source records where each window came from.

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
SPLITS_DIR = REPO_ROOT / "datasets" / "splits"
OUT_DIR = REPO_ROOT / "datasets" / "heldout_inference"
PER_CLASS = 100
TARGET_LEN = 6000
SAMPLING_RATE = 100
P_LEAD_SEC = 10.0
SEED = 42

# Noise is shared across sites so no one sensor stands for the class.
NOISE_PER_SET = {"heldout_AK": 25, "heldout_AM_traffic": 25,
                 "heldout_AM_aircraft": 25, "heldout_AK_train": 25}

# Grouped split of the run that trained compact-v4 / standard-v3 (the Train models)
TRAIN_TEST_MANIFEST = SPLITS_DIR / "split_20260930_144039.csv"
TRAINING_DIR = REPO_ROOT / "notebooks" / "02_labeling" / "labeled_data"


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
        meta["heldout_source"] = "new dates"
        windows += [np.asarray(w, dtype=np.float32) for w in X]
        frames.append(meta)
    return windows, pd.concat(frames, ignore_index=True)


def load_train_from_test():
    """Train windows from the compact-v4 / standard-v3 test split, checked against its train/val groups."""
    man = pd.read_csv(TRAIN_TEST_MANIFEST, low_memory=False)
    test = man[(man.split == "test") & (man.label_name == "Train")]
    assert not set(test.group) & set(man.loc[man.split.isin(["train", "val"]), "group"]), "test group in train/val"
    windows, frames = [], []
    for source_file, rows in test.groupby("source_file"):
        prefix, stamp = Path(source_file).stem.split("_waveforms_")
        meta = pd.read_csv(TRAINING_DIR / f"{prefix}_metadata_{stamp}.csv").loc[rows.source_row]
        X = np.load(TRAINING_DIR / source_file, allow_pickle=True)
        meta = meta.assign(set="test_split_v4", source_file=source_file, source_row=rows.source_row.to_numpy(),
                           group=rows.group.to_numpy(),
                           heldout_source="test split of compact-v4 / standard-v3")
        windows += [np.asarray(X[i], dtype=np.float32) for i in rows.source_row]
        frames.append(meta)
    return windows, pd.concat(frames, ignore_index=True)


def groups_of(meta):
    """Split group of each window, as in the training notebook's grouped split."""
    local_day = pd.to_datetime(meta["start_time"], utc=True).dt.tz_convert("America/Anchorage").dt.date.astype(str)
    event_id = meta["event_id"] if "event_id" in meta.columns else pd.Series(np.nan, index=meta.index)
    from_catalog = meta["label_method"].astype(str).str.startswith("comcat")
    has_event = from_catalog & event_id.notna() & (event_id.astype(str).str.strip() != "")
    return pd.Series(np.where(has_event, "event:" + event_id.astype(str),
                              meta["station"].astype(str) + ":" + local_day), index=meta.index)


def seen_in_training(meta):
    """True for windows whose group or exact (station, start_time) is in any committed split."""
    manifests = sorted(SPLITS_DIR.glob("split_*.csv"))
    if not manifests:
        raise SystemExit(f"no split manifests in {SPLITS_DIR}; cannot check the held-out set")
    used = pd.concat([pd.read_csv(f, usecols=["station", "start_time", "split", "group"], low_memory=False)
                      for f in manifests])
    used = used[used.split.isin(["train", "val", "test"])]
    seen_group = groups_of(meta).isin(set(used.group))
    seen_window = pd.MultiIndex.from_frame(meta[["station", "start_time"]]).isin(
        pd.MultiIndex.from_frame(used[["station", "start_time"]]))
    print(f"checked against {len(manifests)} split manifests: "
          f"{int(seen_group.sum())} windows share a group, {int(seen_window.sum())} match a training window")
    return seen_group | seen_window


def spread(idx, stations, n, rng):
    """Up to n of idx, taken round-robin across stations (each station's windows shuffled)."""
    per_station = [list(rng.permutation(idx[stations == s])) for s in rng.permutation(np.unique(stations))]
    picks = []
    while len(picks) < n and any(per_station):
        for queue in per_station:
            if queue and len(picks) < n:
                picks.append(queue.pop())
    return picks


def picks_of(picks, meta, label):
    return [i for i in picks if meta.label[i] == label]


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
    unseen = ~seen_in_training(meta)
    w_test, m_test = load_train_from_test()
    n_new = len(windows)
    windows += w_test
    meta = pd.concat([meta, m_test], ignore_index=True)
    from_test = meta.index >= n_new
    unseen = np.concatenate([unseen.to_numpy(), np.zeros(len(m_test), dtype=bool)])

    picks = []
    for label, name in LABEL_MAP.items():
        if name == "Noise":
            for set_name, n in NOISE_PER_SET.items():
                pool = meta.index[unseen & (meta.label == label) & (meta.set == set_name)].to_numpy()
                picks += spread(pool, meta.station.to_numpy()[pool], n, rng)
        elif name == "Train":   # unseen passages first, then the v4/v3 test split, spread across its days
            pool = meta.index[unseen & (meta.label == label)].to_numpy()
            picks += list(pool[:PER_CLASS])
            pool = meta.index[from_test].to_numpy()
            picks += spread(pool, meta.group.to_numpy()[pool], PER_CLASS - len(picks_of(picks, meta, label)), rng)
        else:
            pool = meta.index[unseen & (meta.label == label)].to_numpy()
            picks += spread(pool, meta.station.to_numpy()[pool], PER_CLASS, rng)

    X, crop_start = zip(*(crop(windows[i], meta.loc[i].get("p_offset_sec")) for i in picks))
    out = meta.loc[picks].reset_index(drop=True)
    out.insert(0, "subset_id", np.arange(len(out)))
    out["crop_start_sec"] = crop_start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "heldout_waveforms.npy", np.stack(X).astype(np.float32))
    out.to_csv(OUT_DIR / "heldout_metadata.csv", index=False)
    print(out.groupby(["label_name", "station"]).size().to_string())
    print(out.groupby(["label_name", "heldout_source"]).size().to_string())
    print(f"{len(out)} windows -> {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
