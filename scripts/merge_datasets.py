#!/usr/bin/env python3
"""
Merge per-class labeling runs into one training dataset.

Usage (from repo root):
  python scripts/merge_datasets.py \
      notebooks/02_labeling/labeled_data/runs/earthquake_AK_* \
      notebooks/02_labeling/labeled_data/runs/blast_AK_* \
      notebooks/02_labeling/labeled_data/runs/ice_quake_AK_* \
      --cap Earthquake=1500 --out notebooks/02_labeling/labeled_data/dataset_4class

Runs are relabeled by class *name* against configs/classes.json (or
--classes), so runs built at different times line up. Caps drop whole
events, never individual windows, so the event-level split stays clean.
Windows marked for removal in <run>/exclude.txt (one integer index per
line, written by the review notebook) are dropped.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import load_dataset, save_dataset, merge_datasets  # noqa: E402


def apply_exclusions(run_dir: Path) -> Path:
    """If <run>/exclude.txt exists, write a filtered copy and return its path."""
    ex = run_dir / "exclude.txt"
    if not ex.exists():
        return run_dir
    bad = {int(line) for line in ex.read_text().split() if line.strip()}
    X, y, meta, info = load_dataset(run_dir)
    keep = [i for i in range(len(X)) if i not in bad]
    filtered = run_dir.parent / (run_dir.name + "_filtered")
    save_dataset(filtered, [X[i] for i in keep], y[keep], meta.iloc[keep].reset_index(drop=True),
                 info["classes"], info.get("window"), extra={"excluded": sorted(bad)})
    print(f"  {run_dir.name}: dropped {len(bad)} reviewed windows -> {filtered.name}")
    return filtered


def main():
    cfg = json.load(open(REPO_ROOT / "configs" / "classes.json"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--classes", default=None, help="comma-separated override of the class list")
    ap.add_argument("--cap", action="append", default=[], help="Class=N, repeatable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else cfg["classes"]
    caps = {}
    for c in args.cap:
        name, n = c.split("=")
        caps[name.strip()] = int(n)

    runs = []
    for r in args.runs:
        if not (r / "waveforms.npy").exists():
            sys.exit(f"{r} is not a dataset directory")
        runs.append(apply_exclusions(r))

    windows, labels, meta, info = merge_datasets(runs, classes, caps or None, args.seed)
    out = args.out or (REPO_ROOT / "notebooks" / "02_labeling" / "labeled_data"
                       / f"dataset_{len(classes)}class_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dataset(out, windows, labels, meta, classes, info.get("window"),
                 extra={"source_runs": [str(r) for r in runs], "caps": caps})

    y = np.asarray(labels)
    print(f"\nMerged {len(runs)} runs -> {out}")
    for i, c in enumerate(classes):
        n = int((y == i).sum())
        ev = meta.loc[y == i, "event_id"].nunique() if n else 0
        print(f"  {i} {c:12s} {n:6d} windows  {ev:5d} events")
    if "network" in meta.columns:
        print(f"  networks: {meta['network'].value_counts().to_dict()}")
    missing = [c for i, c in enumerate(classes) if (y == i).sum() == 0]
    if missing:
        print(f"\nWARNING: no windows for {missing}. Train with --classes to drop them, or build a run.")


if __name__ == "__main__":
    main()
