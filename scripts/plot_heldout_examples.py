#!/usr/bin/env python3
"""
Plot the held-out inference windows, one figure per class, so the set can be
shown without redistributing waveforms (the .npy stays local; the Raspberry
Shake terms forbid redistributing it).

Each figure shows N_PER_CLASS windows of the class (seeded, spread
round-robin across stations; Train across days), one panel per window
(waveform above spectrogram) titled with its subset_id, station, local time
and how many models that know its class got it right, from
heldout_predictions.csv. Accuracy is reported on the whole set elsewhere.

Usage (from repo root, after make_heldout_subset.py and run_heldout_inference.py):
  python scripts/plot_heldout_examples.py

Output: docs/figures/heldout_<class>.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from plot_label_report_figures import COLORS, panel

SUBSET_DIR = REPO_ROOT / "datasets" / "heldout_inference"
OUT_DIR = REPO_ROOT / "docs" / "figures"
COLORS = {**COLORS, "Earthquake": "#b0413e"}
N_COLS = 10
N_PER_CLASS = 10
SEED = 42


def sample(sub, key, n, rng):
    """Up to n rows of sub, taken round-robin across the values of key (each shuffled)."""
    queues = [list(rng.permutation(g.index)) for _, g in sub.groupby(key)]
    queues = [queues[i] for i in rng.permutation(len(queues))]
    picks = []
    while len(picks) < n and any(queues):
        for q in queues:
            if q and len(picks) < n:
                picks.append(q.pop())
    return sub.loc[picks]


def main():
    X = np.load(SUBSET_DIR / "heldout_waveforms.npy")
    meta = pd.read_csv(SUBSET_DIR / "heldout_metadata.csv")
    pred = pd.read_csv(SUBSET_DIR / "heldout_predictions.csv")
    scored = pred[pred.in_scope].assign(ok=lambda d: d.true == d.predicted)
    n_ok = scored.groupby("subset_id").ok.agg(["sum", "count"])

    rng = np.random.default_rng(SEED)
    meta["day"] = pd.to_datetime(meta.start_time, utc=True).dt.tz_convert("America/Anchorage").dt.date
    for name, sub in meta.groupby("label_name"):
        sub = sample(sub, "day" if name == "Train" else "station", N_PER_CLASS, rng)
        sub = sub.sort_values(["station", "start_time"])
        n_rows = -(-len(sub) // N_COLS)
        fig, axes = plt.subplots(2 * n_rows, N_COLS, figsize=(3.4 * N_COLS, 2.5 * n_rows), squeeze=False,
                                 gridspec_kw={"height_ratios": [1, 1.2] * n_rows, "hspace": 0.55, "wspace": 0.25})
        ymax = np.abs(X[sub.subset_id]).max() if name != "Noise" else None
        for k, (_, m) in enumerate(sub.iterrows()):
            r, c = divmod(k, N_COLS)
            ax_w, ax_s = axes[2 * r, c], axes[2 * r + 1, c]
            w = X[m.subset_id]
            t = pd.Timestamp(m.start_time).tz_convert("America/Anchorage")
            ok, n = n_ok.loc[m.subset_id]
            panel(ax_w, ax_s, w, f"#{m.subset_id} {m.network}.{m.station} {t:%Y-%m-%d %H:%M %Z}\n"
                                 f"{ok}/{n} models correct", COLORS[name])
            lim = ymax if ymax is not None else np.abs(w).max()   # Noise: each site on its own scale
            ax_w.set_ylim(-lim, lim)
            if c == 0:
                ax_s.set_ylabel("Hz", fontsize=7)
            if r == n_rows - 1:
                ax_s.set_xlabel("s", fontsize=7)
            else:
                ax_s.set_xticklabels([])
        for k in range(len(sub), n_rows * N_COLS):   # empty slots in the last row
            r, c = divmod(k, N_COLS)
            axes[2 * r, c].axis("off"); axes[2 * r + 1, c].axis("off")
        out = OUT_DIR / f"heldout_{name.lower()}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: {len(sub)} windows -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
