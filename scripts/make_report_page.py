#!/usr/bin/env python3
"""
Build docs/report.html, the illustrated project report (map, catalogs, CNN
explainer, limitations), from scripts/report_template.html.

The earthquake catalog is every event_id labelled Earthquake in
datasets/splits/*.csv (training) and datasets/heldout_inference/ (held-out),
with time, magnitude, location and depth fetched from USGS ComCat.

Usage (from repo root, needs network):
  python scripts/make_report_page.py
"""

import glob
import io
import json
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "report_template.html"
OUT = REPO_ROOT / "docs" / "report.html"
# configs/ak_events.yaml region padded by 150 km, through the held-out dates
USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
        "&starttime=2020-01-01&endtime=2026-10-01&minmagnitude=3&maxmagnitude=7"
        "&minlatitude=59.6&maxlatitude=62.7&minlongitude=-153.1&maxlongitude=-146.8")


def main():
    train_ids = set()
    for f in glob.glob(str(REPO_ROOT / "datasets" / "splits" / "*.csv")):
        d = pd.read_csv(f)
        train_ids |= set(d.loc[d.label_name == "Earthquake", "event_id"].dropna())
    h = pd.read_csv(REPO_ROOT / "datasets" / "heldout_inference" / "heldout_metadata.csv")
    heldout_ids = set(h.loc[h.label_name == "Earthquake", "event_id"].dropna())

    cat = pd.read_csv(io.StringIO(urlopen(USGS, timeout=60).read().decode()))
    cat = cat.drop_duplicates("id")
    cat["set"] = None
    cat.loc[cat.id.isin(train_ids), "set"] = "train"
    cat.loc[cat.id.isin(heldout_ids), "set"] = "heldout"
    cat = cat[cat.set.notna()].sort_values("time", ascending=False)
    missing = (train_ids | heldout_ids) - set(cat.id)
    if missing:
        print(f"warning: {len(missing)} event ids not found in ComCat: {sorted(missing)[:5]}")

    events = [[r.id, r.time[:16], round(r.mag, 1), round(r.latitude, 3), round(r.longitude, 3),
               round(r.depth, 1), r.place, r.set] for r in cat.itertuples()]
    html = TEMPLATE.read_text().replace("__EVENTS__", json.dumps(events, separators=(",", ":")))
    html = html.replace("__FETCHED__", date.today().isoformat())
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(events)} events "
          f"({(cat.set == 'train').sum()} training, {(cat.set == 'heldout').sum()} held-out)")


if __name__ == "__main__":
    main()
