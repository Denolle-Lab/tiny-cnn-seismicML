#!/usr/bin/env python3
"""
Build the static label-review site under docs/review/ (served by GitHub Pages).

For a sample of stored windows per (class, labeling rule) it renders a small
waveform + spectrogram thumbnail into docs/review/img/ and writes
docs/review/windows.js with the per-window metadata the page needs. The
page (docs/review/index.html) lets a reviewer mark each window Agree
(default) or Disagree, add a note, and download a CSV; scripts/apply_review.py
merges that CSV back into the metadata files.

Only plots and metadata are published: the Raspberry Shake terms forbid
redistributing waveform data, so the .npy files stay local.

Usage (from repo root):
  python scripts/make_review_site.py --train-dir <dir with AK_train_full_*> \
      --n-noise 40 --n-traffic 40
"""

import argparse
import json
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
from src.data.collect import SAMPLING_RATE as SR, WINDOW_SEC, latest_dataset

TZ = ZoneInfo('America/Anchorage')
COLORS = {'Noise': '#2b6ca3', 'Traffic': '#d9822b', 'Train': '#7b3f9e', 'Aircraft': '#3a8f5a'}


def load(data_dir, prefix):
    wf, lf, mf = latest_dataset(data_dir, prefix)
    m = pd.read_csv(mf)
    m['prefix'] = prefix
    m['source_stamp'] = wf.stem.split('_waveforms_')[-1]
    return np.load(wf), m


def thumbnail(w, path, color):
    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(4.8, 2.6), gridspec_kw={'height_ratios': [1, 1.3], 'hspace': 0.12})
    t = np.arange(len(w)) / SR
    ax_w.plot(t, w, lw=0.5, color=color)
    ax_w.set_xlim(0, WINDOW_SEC)
    ax_w.set_xticklabels([])
    ax_w.tick_params(labelsize=6)
    ax_w.set_ylabel('counts', fontsize=6)
    f, tt, S = spectrogram(w, fs=SR, nperseg=256, noverlap=192)
    S = 10 * np.log10(S + 1e-12)
    ax_s.pcolormesh(tt, f, S, shading='auto', cmap='magma', vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    ax_s.set_ylim(0, 50)
    ax_s.set_xlim(0, WINDOW_SEC)
    ax_s.set_ylabel('Hz', fontsize=6)
    ax_s.set_xlabel('s', fontsize=6)
    ax_s.tick_params(labelsize=6)
    fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labeled-dir', default=str(REPO_ROOT / 'notebooks' / '02_labeling' / 'labeled_data'))
    p.add_argument('--traffic-prefixes', nargs='+', default=['AM_romig_traffic_2026-09-09', 'AM_romig_traffic_2026-09-12'])
    p.add_argument('--train-dir', required=True)
    p.add_argument('--train-prefixes', nargs='+', default=['AK_train_full_2026-09-09', 'AK_train_full_2026-09-12'])
    p.add_argument('--n-noise', type=int, default=40, help='Noise windows per Romig station (time of day)')
    p.add_argument('--n-traffic', type=int, default=40, help='Traffic windows per Romig station (time of day)')
    p.add_argument('--n-k222-noise', type=int, default=20)
    p.add_argument('--out', default=str(REPO_ROOT / 'docs' / 'review'))
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    out = Path(args.out)
    (out / 'img').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sets = [load(args.labeled_dir, pre) for pre in args.traffic_prefixes]
    sets += [load(args.train_dir, pre) for pre in args.train_prefixes]

    picks = []  # (X, meta row) selections
    for X, m in sets:
        m = m.copy()
        m['rule'] = np.where(m.label_method.str.contains('burst'), 'night burst',
                             np.where(m.label_method.str.startswith('events'), 'timetable search',
                                      np.where(m.label_method.str.startswith('timeofday'), 'time of day', 'other')))
        for station, g in m.groupby('station'):
            groups = {
                ('Noise', 'time of day'): args.n_noise,
                ('Traffic', 'time of day'): args.n_traffic,
                ('Traffic', 'night burst'): None,           # all
                ('Train', 'timetable search'): None,        # all
                ('Noise', 'timetable search'): args.n_k222_noise,
            }
            for (cls, rule), n in groups.items():
                idx = g.index[(g.label_name == cls) & (g.rule == rule)].to_numpy()
                if len(idx) == 0:
                    continue
                if n is not None and len(idx) > n:
                    idx = np.sort(rng.choice(idx, size=n, replace=False))
                for i in idx:
                    picks.append((X, m.loc[i], cls, rule))

    rows = []
    for k, (X, r, cls, rule) in enumerate(picks):
        wid = f'{r.prefix}__{int(r.window_id)}'
        img = f'img/{wid}.png'
        if not (out / img).exists():
            thumbnail(X[int(r.window_id)], out / img, COLORS.get(cls, '#333'))
        t0 = UTCDateTime(r.start_time)
        local = t0.datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(TZ)
        rows.append({
            'id': wid, 'img': img, 'prefix': r.prefix, 'source_stamp': r.source_stamp, 'window_id': int(r.window_id),
            'network': r.network, 'station': r.station, 'channel': r.channel,
            'start_time_utc': r.start_time, 'local_time': local.strftime('%Y-%m-%d %H:%M %a'),
            'label': int(r.label), 'label_name': cls, 'rule': rule, 'label_method': r.label_method,
            'event_id': None if pd.isna(r.get('event_id', np.nan)) else str(r.get('event_id')),
            'rms': round(float(r.rms), 1), 'rms_band': round(float(r.get('rms_band', np.nan)), 1),
            'band_ratio': round(float(r.band_ratio), 3), 'kurtosis_band': round(float(r.kurtosis_band), 2),
        })
    df = pd.DataFrame(rows)
    (out / 'windows.js').write_text('window.REVIEW_WINDOWS = ' + json.dumps(rows, indent=0) + ';\n')
    df.drop(columns=['img']).to_csv(out / 'windows_metadata.csv', index=False)
    print(df.groupby(['station', 'label_name', 'rule']).size().to_string())
    print(f'{len(df)} windows -> {out}/windows.js, thumbnails in {out}/img/')


if __name__ == '__main__':
    main()
