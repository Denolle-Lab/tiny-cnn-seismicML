#!/usr/bin/env python3
"""
Copy the per-window metadata of collected file sets into datasets/metadata/
so it can be committed: labels, times, stations and per-window features are
the manifest of a dataset, and another machine that runs the same config can
check its pull against it. Waveform files stay local (the Raspberry Shake
terms forbid redistributing them; the archives regenerate them bit for bit).

For every prefix found in the source directories the newest stamp is taken;
the metadata CSV, the summary text and the events report (timetable mode)
are copied. A manifest.csv lists prefix, stamp, station, class counts and
the config that made it.

Usage (from repo root):
  python scripts/export_metadata.py --dirs notebooks/02_labeling/labeled_data /path/to/other/sets
  python scripts/export_metadata.py --check   # compare local sets against the committed metadata
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.collect import latest_dataset

OUT = REPO_ROOT / 'datasets' / 'metadata'


def prefixes_in(d):
    return sorted({f.stem.split('_waveforms_')[0] for f in Path(d).glob('*_waveforms_*.npy')})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dirs', nargs='+', default=[str(REPO_ROOT / 'notebooks' / '02_labeling' / 'labeled_data')])
    p.add_argument('--check', action='store_true',
                   help='Do not copy; compare each local set with the committed metadata (start times and labels)')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.csv'
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    rows = []
    for d in args.dirs:
        for prefix in prefixes_in(d):
            try:
                wf, lf, mf = latest_dataset(d, prefix)
            except FileNotFoundError:  # e.g. the old rule-based windowed_waveforms_* files
                print(f'{prefix}: no matching labels file, skipped')
                continue
            if mf is None:
                print(f'{prefix}: no metadata file, skipped')
                continue
            stamp = wf.stem.split('_waveforms_')[-1]
            meta = pd.read_csv(mf)
            if args.check:
                committed = sorted(OUT.glob(f'{prefix}_metadata_*.csv'))
                if not committed:
                    print(f'{prefix}: not in datasets/metadata')
                    continue
                ref = pd.read_csv(committed[-1])
                same = len(ref) == len(meta) and (ref['start_time'].values == meta['start_time'].values).all() \
                    and (ref['label'].values == meta['label'].values).all()
                print(f'{prefix}: {"identical" if same else "DIFFERS"} ({len(meta)} vs {len(ref)} windows, '
                      f'committed stamp {committed[-1].stem.split("_metadata_")[-1]}, local {stamp})')
                continue
            for old in OUT.glob(f'{prefix}_*'):
                old.unlink()  # one stamp per prefix in the repo
            shutil.copy(mf, OUT / mf.name)
            for extra in (Path(d) / f'{prefix}_summary_{stamp}.txt', Path(d) / f'{prefix}_events_report.csv'):
                if extra.exists():
                    shutil.copy(extra, OUT / extra.name)
            counts = meta['label_name'].value_counts().to_dict()
            rows.append({'prefix': prefix, 'stamp': stamp, 'stations': ' '.join(sorted(meta['station'].unique())),
                         'windows': len(meta), **{f'n_{k.lower()}': v for k, v in counts.items()},
                         'label_methods': ' | '.join(sorted(meta['label_method'].unique()))[:200]})
            print(f'{prefix} {stamp}: {len(meta)} windows {counts}')
    if args.check or not rows:
        return
    new = pd.DataFrame(rows)
    if len(manifest):
        manifest = manifest[~manifest['prefix'].isin(new['prefix'])]
    manifest = pd.concat([manifest, new], ignore_index=True).sort_values('prefix')
    manifest.to_csv(manifest_path, index=False)
    print(f'{len(new)} sets exported; manifest has {len(manifest)} rows -> {manifest_path}')


if __name__ == '__main__':
    main()
