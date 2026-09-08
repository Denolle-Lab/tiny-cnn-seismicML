"""
Window cutting, dataset I/O, event-level splitting and crop expansion.

Pure numpy/scipy: nothing here talks to a data center, so the whole
training side of the pipeline can be unit-tested with synthetic traces.

Dataset layout on disk (one directory per labeling run or per merged
training set)::

    <dir>/waveforms.npy   object array of float32 1-D windows (ragged lengths)
    <dir>/labels.npy      int64 class index per window
    <dir>/metadata.csv    one row per window; must contain 'event_id',
                          'label_name', 'onset_sample', 'network', 'station'
    <dir>/classes.json    {"classes": [...], "window": {...}}

Event windows are longer than the training length (default
[onset-30 s, onset+90 s] = 120 s) so the trainer can cut random 60 s
crops; noise windows are exactly the training length. Both kinds are
bandpassed and z-normalized per window before saving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import signal


# ---------------------------------------------------------------------------
# Single-trace preprocessing
# ---------------------------------------------------------------------------

def preprocess_trace(data: np.ndarray, fs_in: float, fs_out: float = 100.0,
                     lowcut: float = 2.0, highcut: float = 20.0,
                     taper_fraction: float = 0.05) -> np.ndarray:
    """Detrend, taper, bandpass and resample a raw 1-D trace.

    Matches the processing used for the July 2026 AK models (bandpass
    2-20 Hz at 100 Hz) so new classes are comparable with the old ones.
    """
    x = np.asarray(data, dtype=np.float64)
    x = signal.detrend(x, type="linear")
    n = len(x)
    if taper_fraction > 0 and n > 10:
        m = max(1, int(n * taper_fraction))
        win = signal.windows.tukey(n, alpha=2 * m / n)
        x = x * win
    nyq = 0.5 * fs_in
    hi = min(highcut, 0.95 * nyq)
    sos = signal.butter(4, [lowcut / nyq, hi / nyq], btype="band", output="sos")
    x = signal.sosfiltfilt(sos, x)
    if abs(fs_in - fs_out) > 1e-6:
        # resample_poly handles the usual 40/50/200 -> 100 Hz cases exactly
        from fractions import Fraction
        frac = Fraction(fs_out / fs_in).limit_denominator(1000)
        x = signal.resample_poly(x, frac.numerator, frac.denominator)
    return x.astype(np.float32)


def znorm(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    return (w - w.mean()) / (w.std() + 1e-10)


def cut_event_and_noise(trace: np.ndarray, fs: float, onset_sample: int,
                        event_pre_sec: float = 30.0, event_post_sec: float = 90.0,
                        noise_pre_sec: float = 90.0, noise_end_sec: float = 30.0
                        ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Cut the event window and the pre-onset noise window from one trace.

    Returns (event_window, noise_window); either is None if the trace does
    not cover it. Windows are z-normalized. The onset sits
    ``event_pre_sec`` seconds into the event window.
    """
    n = len(trace)
    e0 = onset_sample - int(round(event_pre_sec * fs))
    e1 = onset_sample + int(round(event_post_sec * fs))
    n0 = onset_sample - int(round(noise_pre_sec * fs))
    n1 = onset_sample - int(round(noise_end_sec * fs))

    event = trace[e0:e1] if (e0 >= 0 and e1 <= n) else None
    noise = trace[n0:n1] if (n0 >= 0 and n1 <= n) else None

    if event is not None and (event.std() < 1e-12 or not np.isfinite(event).all()):
        event = None
    if noise is not None and (noise.std() < 1e-12 or not np.isfinite(noise).all()):
        noise = None

    return (znorm(event) if event is not None else None,
            znorm(noise) if noise is not None else None)


def sta_lta_onset(trace: np.ndarray, fs: float, guess_sample: int,
                  search_sec: float = 10.0, sta_sec: float = 0.5,
                  lta_sec: float = 10.0) -> Optional[int]:
    """Refine an onset guess with a classic STA/LTA maximum-ratio pick.

    Looks for the largest STA/LTA within +/- ``search_sec`` of the guess.
    Returns None if the trace is too short around the guess.
    """
    nsta = max(1, int(sta_sec * fs))
    nlta = max(nsta + 1, int(lta_sec * fs))
    lo = int(guess_sample - search_sec * fs)
    hi = int(guess_sample + search_sec * fs)
    if lo - nlta < 0 or hi >= len(trace):
        return None
    x = np.asarray(trace, dtype=np.float64) ** 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(lo, hi)
    sta = (csum[idx + 1] - csum[idx + 1 - nsta]) / nsta
    lta = (csum[idx + 1 - nsta] - csum[idx + 1 - nsta - nlta]) / nlta
    ratio = sta / (lta + 1e-12)
    return int(idx[np.argmax(ratio)])


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def save_dataset(out_dir: Path, windows: Sequence[np.ndarray], labels: Sequence[int],
                 metadata: pd.DataFrame, classes: Sequence[str],
                 window_cfg: Optional[dict] = None, extra: Optional[dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.empty(len(windows), dtype=object)
    for i, w in enumerate(windows):
        X[i] = np.asarray(w, dtype=np.float32)
    np.save(out_dir / "waveforms.npy", X, allow_pickle=True)
    np.save(out_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    info = {"classes": list(classes), "window": window_cfg or {}, "n_windows": int(len(windows))}
    if extra:
        info.update(extra)
    with open(out_dir / "classes.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return out_dir


def load_dataset(in_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    in_dir = Path(in_dir)
    X = np.load(in_dir / "waveforms.npy", allow_pickle=True)
    y = np.load(in_dir / "labels.npy")
    meta = pd.read_csv(in_dir / "metadata.csv")
    with open(in_dir / "classes.json") as f:
        info = json.load(f)
    if not (len(X) == len(y) == len(meta)):
        raise ValueError(f"{in_dir}: waveforms ({len(X)}), labels ({len(y)}) and "
                         f"metadata ({len(meta)}) disagree in length")
    return X, y, meta, info


def merge_datasets(dirs: Iterable[Path], classes: Sequence[str],
                   per_class_cap: Optional[dict] = None, seed: int = 42
                   ) -> tuple[list, list, pd.DataFrame, dict]:
    """Concatenate labeling runs, relabeling by class *name* so runs built
    with different class lists still line up. Optional per-class caps are
    applied at the event level (whole events kept or dropped) so the split
    stays clean.
    """
    rng = np.random.default_rng(seed)
    name_to_idx = {c: i for i, c in enumerate(classes)}
    all_w, all_y, frames = [], [], []
    window_cfg = None
    for d in dirs:
        X, y, meta, info = load_dataset(d)
        if "label_name" not in meta.columns:
            src_classes = info["classes"]
            meta["label_name"] = [src_classes[int(k)] for k in y]
        unknown = set(meta["label_name"]) - set(name_to_idx)
        if unknown:
            raise ValueError(f"{d}: labels {sorted(unknown)} not in class list {list(classes)}")
        meta = meta.copy()
        meta["source_run"] = str(d)
        window_cfg = window_cfg or info.get("window")
        for i in range(len(X)):
            all_w.append(X[i])
            all_y.append(name_to_idx[meta["label_name"].iloc[i]])
        frames.append(meta)
    meta = pd.concat(frames, ignore_index=True)
    y = np.asarray(all_y)

    if per_class_cap:
        keep = np.ones(len(y), dtype=bool)
        for cname, cap in per_class_cap.items():
            ci = name_to_idx[cname]
            idx = np.where(y == ci)[0]
            if len(idx) <= cap:
                continue
            events = meta["event_id"].values[idx]
            uniq = np.unique(events)
            rng.shuffle(uniq)
            kept_events, n = [], 0
            for ev in uniq:
                if n >= cap:
                    break
                kept_events.append(ev)
                n += int((events == ev).sum())
            drop = idx[~np.isin(events, kept_events)]
            keep[drop] = False
        all_w = [w for w, k in zip(all_w, keep) if k]
        y = y[keep]
        meta = meta[keep].reset_index(drop=True)

    meta["label"] = y
    return all_w, list(map(int, y)), meta, {"window": window_cfg}


# ---------------------------------------------------------------------------
# Splitting and cropping
# ---------------------------------------------------------------------------

def _split_groups(uniq: np.ndarray, counts: np.ndarray, ratios, rng) -> dict:
    """Shuffle groups and cut them into 3 splits by cumulative window count,
    forcing at least one group into val and test when there are >= 3 groups."""
    order = rng.permutation(len(uniq))
    uniq, counts = uniq[order], counts[order]
    cum = np.cumsum(counts) / counts.sum()
    split = np.where(cum <= ratios[0], 0, np.where(cum <= ratios[0] + ratios[1], 1, 2))
    if len(uniq) >= 3:
        for s in (1, 2):
            if not (split == s).any():
                split[-s] = s
    return dict(zip(uniq.tolist(), split.tolist()))


def event_level_split(event_ids: Sequence, ratios=(0.7, 0.15, 0.15), seed: int = 42,
                      group_key: Optional[Sequence] = None,
                      strata: Optional[Sequence] = None) -> dict:
    """Assign every window to train/val/test so that all windows of one event
    (all stations, the event window and its noise window) land in the same
    split. Returns {"train": mask, "val": mask, "test": mask}.

    ``group_key`` overrides the grouping (e.g. event_id + network).
    ``strata`` (per-window labels) makes the split stratified at the event
    level: each event is assigned to the highest label it carries (its
    event class, since Noise is 0), and each class is split separately, so
    a small class still appears in val and test.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    keys = np.asarray(group_key if group_key is not None else event_ids)
    rng = np.random.default_rng(seed)
    event_to_split: dict = {}

    if strata is None:
        uniq, counts = np.unique(keys, return_counts=True)
        event_to_split = _split_groups(uniq, counts, ratios, rng)
    else:
        strata = np.asarray(strata)
        df = pd.DataFrame({"key": keys, "label": strata})
        per_event = df.groupby("key")["label"].agg(["max", "size"])
        for _, sub in per_event.groupby("max"):
            event_to_split.update(_split_groups(sub.index.to_numpy(), sub["size"].to_numpy(), ratios, rng))

    assign = np.array([event_to_split[k] for k in keys.tolist()])
    return {"train": assign == 0, "val": assign == 1, "test": assign == 2}


def expand_crops(X: Sequence[np.ndarray], y: Sequence[int], onset_samples: Sequence[int],
                 target_len: int = 6000, n_onset_crops: int = 1, n_free_crops: int = 1,
                 seed: int = 42, renormalize: bool = True
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn ragged windows into fixed-length training segments.

    Windows already of ``target_len`` pass through once. Longer windows get
    ``n_onset_crops`` crops that contain the onset plus ``n_free_crops``
    crops from anywhere. Returns (X_fixed, y_fixed, source_index) so callers
    can trace a crop back to its window. Call this *after* splitting.
    """
    rng = np.random.default_rng(seed)
    out_x, out_y, src = [], [], []
    for i, w in enumerate(X):
        w = np.asarray(w, dtype=np.float32)
        slack = len(w) - target_len
        if slack < 0:
            continue  # too short; skip rather than pad
        if slack == 0:
            starts = [0]
        else:
            p = int(onset_samples[i]) if onset_samples[i] is not None and not np.isnan(onset_samples[i]) else 0
            onset_hi = int(min(max(p, 0), slack))
            starts = list(rng.integers(0, onset_hi + 1, size=n_onset_crops)) + \
                     list(rng.integers(0, slack + 1, size=n_free_crops))
        for s in starts:
            c = w[s:s + target_len]
            out_x.append(znorm(c) if renormalize else c)
            out_y.append(int(y[i]))
            src.append(i)
    if not out_x:
        return (np.zeros((0, target_len), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64))
    return np.stack(out_x), np.asarray(out_y, dtype=np.int64), np.asarray(src, dtype=np.int64)


def class_weights_inverse_freq(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    if w.sum() > 0:
        w = w / w.sum() * num_classes
    return w.astype(np.float32)
