#!/usr/bin/env python3
"""
Count ComCat events by type in a region. Run this before promising a class.

Usage (from repo root):
  python scripts/inventory_event_types.py                       # Anchorage box, since 2015
  python scripts/inventory_event_types.py --radius-km 400 --start 2010-01-01
  python scripts/inventory_event_types.py --types "ice quake,landslide,avalanche"

Prints a table of counts per event type and magnitude floor, and writes
the same table to stdout as CSV when --csv is given.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.catalog import count_events_by_type, _region_bounds  # noqa: E402

DEFAULT_TYPES = ["earthquake", "quarry blast", "explosion", "ice quake", "landslide",
                 "avalanche", "volcanic eruption", "sonic boom", "mine collapse", "other event"]


def main():
    cfg = json.load(open(REPO_ROOT / "configs" / "classes.json"))
    region = cfg["region"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--center-lat", type=float, default=region["center_lat"])
    ap.add_argument("--center-lon", type=float, default=region["center_lon"])
    ap.add_argument("--radius-km", type=float, default=region["max_station_distance_km"] + 100)
    ap.add_argument("--min-magnitudes", default="0,1,2,2.5,3")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES))
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    bounds = _region_bounds(args.center_lat, args.center_lon, args.radius_km)
    mags = [float(m) for m in args.min_magnitudes.split(",")]
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    print(f"Region: {args.radius_km:.0f} km box around ({args.center_lat}, {args.center_lon}), "
          f"from {args.start} to {args.end or 'now'}\n")
    header = ["eventtype"] + [f"M>={m:g}" for m in mags]
    rows = []
    for et in types:
        row = [et]
        for m in mags:
            try:
                row.append(str(count_events_by_type(et, args.start, args.end, m, **bounds)))
            except Exception as exc:  # noqa: BLE001
                row.append(f"err:{type(exc).__name__}")
        rows.append(row)
        print("  ".join(f"{c:>14s}" if i else f"{c:18s}" for i, c in enumerate(row)))

    if args.csv:
        print("\n" + ",".join(header))
        for r in rows:
            print(",".join(r))


if __name__ == "__main__":
    main()
