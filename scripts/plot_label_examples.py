#!/usr/bin/env python3
"""
Example windows for every label, from the training sets in labeled_data/:
waveform and spectrogram per window, one row per class, drawn with the
label-report panel so the figures match.

Signal rows share one amplitude scale per row. The Noise row holds one window
from each site (AK broadband, Romig, K222, Turnagain) plus one random Noise
window, each on its own scale, since the sensors differ by orders of magnitude.
Earthquake windows (120 s) are cut to 60 s starting 10 s before P, as in the
held-out subset.

Usage (from repo root):
  python scripts/plot_label_examples.py

Output: docs/figures/label_examples.png
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

LABELED_DIR = REPO_ROOT / "notebooks" / "02_labeling" / "labeled_data"
OUT = REPO_ROOT / "docs" / "figures" / "label_examples.png"
COLORS = {**COLORS, "Earthquake": "#b0413e"}
SR, WIN, P_LEAD_SEC, N = 100, 6000, 10.0, 5
SEED = 42

SITES = {"AK": "AK broadband", "AM_romig_traffic_*": "Romig", "AK_train_*": "K222", "AM_aircraft_*": "Turnagain"}


def load_sets(pattern):
    """Every file set matching pattern; earthquakes cropped to 60 s."""
    X, frames = [], []
    for wf in sorted(LABELED_DIR.glob(f"{pattern}_waveforms_*.npy")):
        prefix, stamp = wf.stem.split("_waveforms_")
        m = pd.read_csv(LABELED_DIR / f"{prefix}_metadata_{stamp}.csv")
        p_offsets = m["p_offset_sec"] if "p_offset_sec" in m else pd.Series(np.nan, index=m.index)
        for w, p_off in zip(np.load(wf, allow_pickle=True), p_offsets):
            w = np.asarray(w, dtype=np.float32)
            if len(w) > WIN:
                s = min(max(int(round((p_off - P_LEAD_SEC) * SR)), 0), len(w) - WIN)
                w = w[s:s + WIN]
            X.append(w)
        frames.append(m)
    return np.stack(X), pd.concat(frames, ignore_index=True)


def title(m, w, site=None):
    """Station, local time, and rms of the plotted (filtered, cropped) trace."""
    t = pd.Timestamp(m.start_time).tz_convert("America/Anchorage")
    head = f"{site}: " if site else ""
    return f"{head}{m.network}.{m.station} {t:%Y-%m-%d %H:%M %Z}\nrms {np.std(w):.1f} counts"


def main():
    rng = np.random.default_rng(SEED)
    sets = {pat: load_sets(pat) for pat in SITES}
    Xe, me = sets["AK"]
    Xr, mr = sets["AM_romig_traffic_*"]
    Xk, mk = sets["AK_train_*"]
    Xa, ma = sets["AM_aircraft_*"]

    def draw(sub):
        return sorted(rng.choice(sub.index.to_numpy(), N, replace=False))

    daytime = ~mr.label_method.str.contains("burst")
    rows = [
        ("Earthquake", [(Xe, me, i, None) for i in draw(me[me.label_name == "Earthquake"])]),
        ("Traffic", [(Xr, mr, i, None) for i in draw(mr[(mr.label_name == "Traffic") & daytime])]),
        ("Train", [(Xk, mk, i, None) for i in draw(mk[mk.label_name == "Train"])]),
        ("Aircraft", [(Xa, ma, i, None) for i in draw(ma[ma.label_name == "Aircraft"])]),
    ]
    noise = []
    for pat, site in SITES.items():
        X, m = sets[pat]
        noise.append((X, m, rng.choice(m.index[m.label_name == "Noise"]), site))
    pat = rng.choice(list(SITES))
    X, m = sets[pat]
    taken = {i for Xn, mn, i, _ in noise if mn is m}
    noise.append((X, m, rng.choice(m.index[(m.label_name == "Noise") & ~m.index.isin(taken)]), f"random ({SITES[pat]})"))
    rows.append(("Noise", noise))

    fig, axes = plt.subplots(2 * len(rows), N, figsize=(3.4 * N, 2.5 * len(rows)),
                             gridspec_kw={"height_ratios": [1, 1.2] * len(rows), "hspace": 0.5, "wspace": 0.25})
    for r, (name, panels) in enumerate(rows):
        row_max = max(np.abs(X[i]).max() for X, _, i, _ in panels)
        for c, (X, m, i, site) in enumerate(panels):
            ax_w, ax_s = axes[2 * r, c], axes[2 * r + 1, c]
            panel(ax_w, ax_s, X[i], title(m.loc[i], X[i], site), COLORS[name])
            ymax = np.abs(X[i]).max() if name == "Noise" else row_max
            ax_w.set_ylim(-ymax, ymax)
            if c == 0:
                ax_w.set_ylabel(name, fontsize=9, fontweight="bold")
                ax_s.set_ylabel("Hz", fontsize=7)
            if r == len(rows) - 1:
                ax_s.set_xlabel("s", fontsize=7)
            else:
                ax_s.set_xticklabels([])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
