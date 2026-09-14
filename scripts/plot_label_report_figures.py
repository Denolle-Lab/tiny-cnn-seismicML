#!/usr/bin/env python3
"""
Figures for the label-verification report (docs/reports/label_verification_<date>.html).

Reads collected file sets (metadata CSV + stored 60 s windows) and draws:
  1. class galleries: random stored windows per (class, labeling rule), waveform
     and spectrogram, rms in the title, one amplitude scale per row
  2. train detection context: 5-30 Hz rms per minute around every scheduled
     passage, scheduled time and detected windows marked
  3. full-day 5-30 Hz rms at the train station with detections
  4. sensor comparison: diurnal 5-30 Hz rms at a strong-motion airfield station,
     a Shake under the departure path, and the Romig Shake

Usage (from repo root; paths default to the Romig sets in labeled_data/ and
the K222 sets in --train-dir):
  python scripts/plot_label_report_figures.py --train-dir /path/with/AK_train_full_* --out docs/figures
"""

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from scipy.signal import spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.collect import SAMPLING_RATE as SR, WINDOW_SEC, latest_dataset, local_day_bounds, local_hour

TZ = 'America/Anchorage'
COLORS = {'Noise': '#2b6ca3', 'Traffic': '#d9822b', 'Train': '#7b3f9e', 'Aircraft': '#3a8f5a'}


def load(data_dir, prefix):
    wf, lf, mf = latest_dataset(data_dir, prefix)
    return np.load(wf), np.load(lf), pd.read_csv(mf)


def panel(ax_w, ax_s, w, title, color):
    t = np.arange(len(w)) / SR
    ax_w.plot(t, w, lw=0.4, color=color)
    ax_w.set_xlim(0, WINDOW_SEC)
    ax_w.set_xticklabels([])
    ax_w.tick_params(labelsize=6)
    ax_w.set_title(title, fontsize=7)
    f, tt, S = spectrogram(w, fs=SR, nperseg=256, noverlap=192)
    S = 10 * np.log10(S + 1e-12)
    ax_s.pcolormesh(tt, f, S, shading='auto', cmap='magma', vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    ax_s.set_ylim(0, 50)
    ax_s.set_xlim(0, WINDOW_SEC)
    ax_s.tick_params(labelsize=6)


def gallery(groups, out, n=5, seed=0):
    """groups: list of (row title, X, meta subset). One row per group, n random windows each."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2 * len(groups), n, figsize=(3.4 * n, 2.5 * len(groups)),
                             gridspec_kw={'height_ratios': [1, 1.2] * len(groups), 'hspace': 0.42, 'wspace': 0.25},
                             squeeze=False)
    for r, (title, X, sub, color) in enumerate(groups):
        idx = sub.index.to_numpy()
        pick = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else []
        ymax = max((np.abs(X[i]).max() for i in pick), default=1.0)
        for c in range(n):
            ax_w, ax_s = axes[2 * r, c], axes[2 * r + 1, c]
            if c >= len(pick):
                ax_w.axis('off'); ax_s.axis('off'); continue
            i = pick[c]
            m = sub.loc[i]
            hl = local_hour(UTCDateTime(m.start_time), TZ)
            panel(ax_w, ax_s, X[i], f'{m.station} {m.start_time[:10]} {int(hl):02d}:{int((hl % 1) * 60):02d} local\n'
                                    f'rms {m.rms:.0f}, band ratio {m.band_ratio:.2f}', color)
            ax_w.set_ylim(-ymax, ymax)
            if c == 0:
                ax_w.set_ylabel(title, fontsize=8, fontweight='bold')
                ax_s.set_ylabel('Hz', fontsize=7)
            if r == len(groups) - 1:
                ax_s.set_xlabel('s', fontsize=7)
            else:
                ax_s.set_xticklabels([])
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labeled-dir', default=str(REPO_ROOT / 'notebooks' / '02_labeling' / 'labeled_data'))
    p.add_argument('--traffic-prefixes', nargs='+', default=['AM_romig_traffic_2026-09-09', 'AM_romig_traffic_2026-09-12'])
    p.add_argument('--train-dir', required=True)
    p.add_argument('--train-prefixes', nargs='+', default=['AK_train_full_2026-09-09', 'AK_train_full_2026-09-12'])
    p.add_argument('--train-all-prefixes', nargs='+', default=['AK_k222_all_2026-09-09', 'AK_k222_all_2026-09-12'],
                   help='Same station-days collected with --label-from all (every minute kept) for the rms curves')
    p.add_argument('--sensor-dir', default=None, help='Directory with the K204 / RFD97 rule-mode test sets')
    p.add_argument('--out', default=str(REPO_ROOT / 'docs' / 'figures'))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- traffic sets (Romig)
    Xt, yt, mt = [], [], []
    for pre in args.traffic_prefixes:
        X, y, m = load(args.labeled_dir, pre)
        Xt.append(X); yt.append(y); mt.append(m)
    Xt = np.concatenate(Xt); mt = pd.concat(mt, ignore_index=True)
    # ---- train sets (K222)
    Xk, mk, reports = [], [], []
    for pre in args.train_prefixes:
        X, y, m = load(args.train_dir, pre)
        Xk.append(X); mk.append(m)
        rp = Path(args.train_dir) / f'{pre}_events_report.csv'
        if rp.exists():
            reports.append(pd.read_csv(rp))
    Xk = np.concatenate(Xk); mk = pd.concat(mk, ignore_index=True)
    ma = pd.concat([load(args.train_dir, pre)[2] for pre in args.train_all_prefixes], ignore_index=True)
    ma['t'] = ma.start_time.apply(UTCDateTime)
    labeled_train = set(mk.loc[mk.label_name == 'Train', 'start_time'])
    ma['label_name'] = np.where(ma.start_time.isin(labeled_train), 'Train', 'Noise')

    # ---- Figure 1: class galleries
    is_burst = mt['label_method'].str.contains('burst')
    groups = [
        ('Noise\n(01-05 local)', Xt, mt[(mt.label_name == 'Noise')], COLORS['Noise']),
        ('Traffic\n(07-19 local)', Xt, mt[(mt.label_name == 'Traffic') & ~is_burst], COLORS['Traffic']),
        ('Traffic\n(night burst)', Xt, mt[(mt.label_name == 'Traffic') & is_burst], COLORS['Traffic']),
        ('Train\n(K222, timetable)', Xk, mk[mk.label_name == 'Train'], COLORS['Train']),
        ('Noise at K222\n(outside spans)', Xk, mk[mk.label_name == 'Noise'], COLORS['Noise']),
    ]
    gallery(groups, out / 'report_class_gallery.png', n=5)

    # ---- Figure 2: train detection context
    ev = pd.read_csv(REPO_ROOT / 'configs' / 'events' / 'arr_summer_2026.csv')
    ev = ev[ev.station == 'K222'].copy()
    ev['t'] = ev.time.apply(UTCDateTime)
    mk['t'] = mk.start_time.apply(UTCDateTime)
    days = sorted({pre.rsplit('_', 1)[-1] for pre in args.train_prefixes})  # local days from the prefixes
    hits = []
    for d in days:
        d0, d1 = local_day_bounds(d, TZ)
        for e in ev[(ev.t >= d0) & (ev.t < d1)].itertuples():
            hits.append((d, e))
    ncol = 4
    nrow = int(np.ceil(len(hits) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)
    for k, (d, e) in enumerate(hits):
        ax = axes[k // ncol, k % ncol]
        span = ma[(ma.t >= e.t - 900) & (ma.t <= e.t + 900)]
        if span.empty:
            ax.set_title(f'{e.event_id}: no data', fontsize=7); continue
        rel = np.array([float(x - e.t) / 60 for x in span.t])
        ax.bar(rel, span.rms_band, width=0.9, color=[COLORS['Train'] if l == 'Train' else '#bbbbbb' for l in span.label_name])
        med = np.median(span.rms_band)
        ax.axhline(3 * med, color='k', lw=0.7, ls=':', label='3 x span median')
        ax.axvline(0, color='k', lw=0.8, ls='--', label='timetable estimate')
        hl = local_hour(e.t, TZ)
        ax.set_title(f'{e.event_id.replace("_" + d, "")}\n{d} est. {int(hl):02d}:{int((hl % 1) * 60):02d} local', fontsize=7)
        ax.set_xlabel('minutes from estimate', fontsize=7)
        ax.tick_params(labelsize=6)
        if k % ncol == 0:
            ax.set_ylabel('5-30 Hz rms (counts)', fontsize=7)
        if k == 0:
            ax.legend(fontsize=6, loc='upper left')
    for k in range(len(hits), nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')
    fig.suptitle('K222: every scheduled passage, +/- 15 min of 5-30 Hz rms per minute. Purple = labeled Train; grey = rest of the span, dropped as ambiguous', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'report_train_detection_context.png', dpi=120)
    plt.close(fig)

    # ---- Figure 3: full days at K222
    fig, axes = plt.subplots(1, len(days), figsize=(6.5 * len(days), 3.6), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], days):
        d0, d1 = local_day_bounds(d, TZ)
        sub = ma[(ma.t >= d0) & (ma.t < d1)]
        h = np.array([local_hour(x, TZ) for x in sub.t])
        ax.plot(h, sub.rms_band, lw=0.6, color='#777')
        tr = sub[sub.label_name == 'Train']
        ax.scatter([local_hour(x, TZ) for x in tr.t], tr.rms_band, s=22, color=COLORS['Train'], zorder=3, label='Train')
        for e in ev[(ev.t >= d0) & (ev.t < d1)].itertuples():
            ax.axvline(local_hour(e.t, TZ), color='k', lw=0.6, ls='--')
        ax.set_yscale('log')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_title(f'K222 {d}: 5-30 Hz rms per minute; dashed = timetable estimates', fontsize=9)
        ax.set_xlabel(f'local hour ({TZ})')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel('rms (counts)')
    fig.tight_layout()
    fig.savefig(out / 'report_train_full_days.png', dpi=120)
    plt.close(fig)

    # ---- Figure 4: sensor comparison (diurnal 5-30 Hz rms)
    if args.sensor_dir:
        sets = [('AK.K204 (airfield, accelerometer)', args.sensor_dir, 'AK_aircraft_test2_2026-09-09', 'K204'),
                ('AM.RFD97 (Turnagain, geophone)', args.sensor_dir, 'AM_aircraft_test_2026-09-09', 'RFD97'),
                ('AM.R1796 (Romig, geophone)', args.labeled_dir, 'AM_romig_traffic_2026-09-09', 'R1796'),
                ('AK.K222 (Potter, accelerometer)', args.train_dir, 'AK_k222_all_2026-09-09', 'K222')]
        fig, ax = plt.subplots(figsize=(11, 3.8))
        for name, d, pre, sta in sets:
            try:
                _, _, m = load(d, pre)
            except FileNotFoundError:
                continue
            m = m[m.station == sta]
            col = 'rms_band' if 'rms_band' in m.columns else 'rms'
            h = np.array([local_hour(UTCDateTime(x), TZ) for x in m.start_time])
            o = np.argsort(h)
            h, v = h[o], m[col].to_numpy()[o].astype(float)
            v[np.r_[False, np.diff(h) > 2.5 / 60]] = np.nan  # break the line across dropped hours
            ax.plot(h, v, lw=0.6, label=name)
        ax.set_yscale('log')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel(f'local hour ({TZ}), 2026-09-09')
        ax.set_ylabel('5-30 Hz rms (counts)')
        ax.set_title('Sensor comparison: the airfield accelerometer sits at its self-noise floor all day', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'report_sensor_comparison.png', dpi=120)
        plt.close(fig)

    # ---- counts table for the report
    rows = []
    for name, m in (('traffic', mt), ('train', mk)):
        m = m.copy()
        m['day'] = [str((UTCDateTime(x).datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo(TZ))).date())
                    for x in m.start_time]  # local calendar day
        m['rule'] = np.where(m.label_method.str.contains('burst'), 'timeofday+burst',
                             m.label_method.str.split(':').str[0])
        rows.append(m.groupby(['day', 'station', 'label_name', 'rule']).size().rename('windows').reset_index())
    counts = pd.concat(rows, ignore_index=True)
    counts.to_csv(out / 'report_label_counts.csv', index=False)
    print(counts.to_string(index=False))
    if reports:
        rep = pd.concat(reports, ignore_index=True)
        rep.to_csv(out / 'report_train_detections.csv', index=False)
        print(rep.to_string(index=False))
    print(f'figures in {out}/')


if __name__ == '__main__':
    main()
