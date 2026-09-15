#!/usr/bin/env python3
"""
Merge a CSV downloaded from the review site (docs/review/) into the metadata
files it came from: sets ``reviewed`` to the reviewer name, ``review_label``
to the window's label name for Agree and to ``disagree`` for Disagree, and
``review_note`` / ``review_time``. Rows are matched by (prefix, source_stamp,
window_id), so the CSV only applies to the exact file sets the site was built
from; sets with another stamp are reported and skipped.

Usage (from repo root):
  python scripts/apply_review.py label_review_marine_2026-09-15.csv \
      --dirs notebooks/02_labeling/labeled_data /path/with/AK_train_full_*
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv')
    p.add_argument('--dirs', nargs='+', default=['notebooks/02_labeling/labeled_data'],
                   help='Directories holding the <prefix>_metadata_<stamp>.csv files')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    rev = pd.read_csv(args.csv)
    need = {'prefix', 'source_stamp', 'window_id', 'decision', 'reviewer', 'label_name'}
    if not need <= set(rev.columns):
        raise SystemExit(f'{args.csv} is missing columns {sorted(need - set(rev.columns))}')
    for (prefix, stamp), g in rev.groupby(['prefix', 'source_stamp']):
        paths = [Path(d) / f'{prefix}_metadata_{stamp}.csv' for d in args.dirs]
        path = next((x for x in paths if x.exists()), None)
        if path is None:
            print(f'{prefix} {stamp}: metadata file not found in {args.dirs}, skipped ({len(g)} rows)')
            continue
        meta = pd.read_csv(path)
        for col in ('reviewed', 'review_label', 'review_note', 'review_time'):
            if col not in meta.columns:
                meta[col] = ''
        meta = meta.set_index('window_id')
        g = g.set_index('window_id')
        common = g.index.intersection(meta.index)
        meta.loc[common, 'reviewed'] = g.loc[common, 'reviewer'].astype(str)
        meta.loc[common, 'review_label'] = [lab if d == 'agree' else 'disagree'
                                            for lab, d in zip(g.loc[common, 'label_name'], g.loc[common, 'decision'])]
        meta.loc[common, 'review_note'] = g.loc[common, 'note'].fillna('').astype(str) if 'note' in g else ''
        meta.loc[common, 'review_time'] = g.loc[common, 'reviewed_at'].astype(str) if 'reviewed_at' in g else ''
        n_dis = int((g.loc[common, 'decision'] == 'disagree').sum())
        print(f'{path.name}: {len(common)} reviewed, {n_dis} disagree' + (' (dry run)' if args.dry_run else ''))
        if not args.dry_run:
            meta.reset_index().to_csv(path, index=False)


if __name__ == '__main__':
    main()
