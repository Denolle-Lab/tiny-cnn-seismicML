#!/usr/bin/env python3
"""
Build a labeled window dataset for ONE class from verified catalog events.

Generalizes notebooks/02_labeling/download_AK_only_data.ipynb: the class
is selected by USGS ComCat ``eventtype`` (configs/classes.json maps class
names to event types), and every event window comes with a pre-onset
noise window from the same trace.

Usage (from repo root):
  python scripts/build_dataset.py --class Earthquake --network AK --max-events 300
  python scripts/build_dataset.py --class Blast      --network AK --start 2015-01-01
  python scripts/build_dataset.py --class "Ice quake" --network AK --radius-km 300
  python scripts/build_dataset.py --class Earthquake --network AM   # Raspberry Shakes

Each run writes notebooks/02_labeling/labeled_data/runs/<class>_<network>_<ts>/
with waveforms.npy, labels.npy, metadata.csv, classes.json, run_config.json.
Merge runs with scripts/merge_datasets.py before training.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.windows import save_dataset  # noqa: E402
from src.data import catalog as cat  # noqa: E402


def main():
    cfg = json.load(open(REPO_ROOT / "configs" / "classes.json"))
    classes = cfg["classes"]
    region = cfg["region"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="class_name", required=True,
                    help=f"Class to build (one of {classes[1:]} or any key in classes.json 'sources')")
    ap.add_argument("--network", default="AK", help="AK (IRIS broadband) or AM (Raspberry Shake)")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-magnitude", type=float, default=None, help="override classes.json")
    ap.add_argument("--max-magnitude", type=float, default=None)
    ap.add_argument("--center-lat", type=float, default=region["center_lat"])
    ap.add_argument("--center-lon", type=float, default=region["center_lon"])
    ap.add_argument("--radius-km", type=float, default=region["max_station_distance_km"],
                    help="event search box half-width; stations are matched within --max-station-km of each event")
    ap.add_argument("--max-station-km", type=float, default=region["max_station_distance_km"])
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--max-stations-per-event", type=int, default=6)
    ap.add_argument("--onset", choices=["auto", "pick", "taup"], default="auto")
    ap.add_argument("--no-stalta", action="store_true", help="do not refine onsets with STA/LTA")
    ap.add_argument("--no-noise", action="store_true", help="do not cut pre-onset noise windows")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.class_name not in cfg["sources"]:
        sys.exit(f"Class '{args.class_name}' has no source in configs/classes.json. "
                 f"Known: {list(cfg['sources'])}")
    if args.class_name in classes:
        class_index = classes.index(args.class_name)
    else:
        # Not in the canonical list yet: still allowed, merge_datasets relabels by name
        class_index = len(classes)
        classes = classes + [args.class_name]
    noise_index = classes.index("Noise")

    src = cfg["sources"][args.class_name]
    min_mag = args.min_magnitude if args.min_magnitude is not None else src["min_magnitude"]
    max_mag = args.max_magnitude if args.max_magnitude is not None else src["max_magnitude"]
    bounds = cat._region_bounds(args.center_lat, args.center_lon, args.radius_km)
    wcfg = cat.WindowConfig.from_dict(cfg["window"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.class_name.lower().replace(" ", "_")
    out = args.out or (REPO_ROOT / "notebooks" / "02_labeling" / "labeled_data" / "runs"
                       / f"{slug}_{args.network}_{ts}")

    print(f"Class {args.class_name!r} (index {class_index}) from ComCat eventtype={src['eventtype']} "
          f"M{min_mag}-{max_mag}, {args.start} to {args.end or 'now'}")
    t0 = time.time()
    events = cat.get_events_by_type(src["eventtype"], args.start, args.end, min_mag, max_mag,
                                    bounds, args.max_events)
    print(f"  {len(events)} events ({time.time() - t0:.0f} s)")
    if len(events) == 0:
        sys.exit("No events found; widen --radius-km, lower --min-magnitude, or change --start.")

    stations = cat.get_stations(args.network, args.start,
                                cat._region_bounds(args.center_lat, args.center_lon,
                                                   args.radius_km + args.max_station_km))
    print(f"  {len(stations)} {args.network} stations with a vertical channel")
    if stations.empty:
        sys.exit("No stations found.")

    windows, labels, rows = cat.collect_windows(
        events, stations, args.class_name, class_index, noise_index, args.network, wcfg,
        max_station_km=args.max_station_km, max_stations_per_event=args.max_stations_per_event,
        onset_mode=args.onset, stalta_refine=not args.no_stalta, noise_per_event=not args.no_noise)

    if not windows:
        sys.exit("No windows collected. Check station coverage and onset settings.")

    meta = pd.DataFrame(rows)
    run_cfg = dict(vars(args), out=str(out), classes=classes, eventtype=src["eventtype"],
                   min_magnitude=min_mag, max_magnitude=max_mag, n_events=len(events),
                   built=ts, window=cat.window_config_dict(wcfg))
    run_cfg["class_name"] = args.class_name
    save_dataset(out, windows, labels, meta, classes, cat.window_config_dict(wcfg),
                 extra={"run_config": run_cfg})
    with open(out / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    counts = meta["label_name"].value_counts().to_dict()
    print(f"\nSaved {len(windows)} windows to {out}")
    print(f"  per class: {counts}")
    print(f"  onset sources: {meta[meta.window_type == 'event']['onset_source'].value_counts().to_dict()}")
    print(f"  events with data: {meta['event_id'].nunique()} / {len(events)}")
    print("\nNext: review with notebooks/02_labeling/review_windows.ipynb (or your own viewer), "
          "then merge runs with scripts/merge_datasets.py")


if __name__ == "__main__":
    main()
