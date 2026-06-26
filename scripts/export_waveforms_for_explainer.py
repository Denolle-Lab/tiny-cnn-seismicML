#!/usr/bin/env python3
"""
Export a subset of labeled AK waveforms to JSON for the CNN Explainer frontend.
Run from repo root: python scripts/export_waveforms_for_explainer.py
Output: explainer-app/public/waveforms.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELED_DIR = REPO_ROOT / "notebooks" / "02_labeling" / "labeled_data"
OUTPUT = REPO_ROOT / "explainer-app" / "public" / "waveforms.json"

def main():
    waveform_files = sorted(LABELED_DIR.glob("AK_waveforms_*.npy"))
    label_files = sorted(LABELED_DIR.glob("AK_labels_*.npy"))
    meta_files = sorted(LABELED_DIR.glob("AK_metadata_*.csv"))
    if not waveform_files or not label_files:
        print("No AK_waveforms_*.npy / AK_labels_*.npy found in", LABELED_DIR)
        return
    waveforms_file = waveform_files[-1]
    labels_file = label_files[-1]
    metadata_file = meta_files[-1] if meta_files else None

    X = np.load(waveforms_file)  # (N, 6000) or (N, 1, 6000)
    y = np.load(labels_file)
    if X.ndim == 3:
        X = X[:, 0, :]
    metadata = pd.read_csv(metadata_file) if metadata_file is not None else None

    # Build index by label
    noise_idx = np.where(y == 0)[0]
    eq_idx = np.where(y == 2)[0]

    # Pick a few noise samples
    n_noise = min(3, len(noise_idx))
    noise_choices = np.random.default_rng(42).choice(noise_idx, size=n_noise, replace=False)

    # Earthquake by magnitude bins: small (<3), medium (3–4), large (>=4)
    out = {
        "sampling_rate_hz": 100,
        "length": int(X.shape[1]),
        "noise": [],
        "earthquake": {"small": [], "medium": [], "large": []}
    }

    # Noise: include station if we have metadata (noise rows may have station in a different table)
    for i in noise_choices:
        rec = {"waveform": X[i].tolist(), "label": "Noise", "index": int(i)}
        if metadata is not None and i < len(metadata):
            row = metadata.iloc[i]
            if row.get("label", 0) == 0 and "station" in row:
                rec["station"] = str(row.get("station", ""))
                rec["network"] = str(row.get("network", ""))
        out["noise"].append(rec)

    if metadata is not None and "magnitude" in metadata.columns:
        meta = metadata.copy()
        meta["row_index"] = np.arange(len(meta))
        eq_meta = meta[meta["label"] == 2]
        small = eq_meta[eq_meta["magnitude"] < 3.0]
        medium = eq_meta[(eq_meta["magnitude"] >= 3.0) & (eq_meta["magnitude"] < 4.0)]
        large = eq_meta[eq_meta["magnitude"] >= 4.0]
        for name, df in [("small", small), ("medium", medium), ("large", large)]:
            n_pick = min(2, len(df))
            if n_pick == 0:
                continue
            indices = df["row_index"].values
            chosen = np.random.default_rng(42).choice(indices, size=n_pick, replace=False)
            for idx in chosen:
                row = meta.iloc[idx]
                # P at window start (0); S and surface estimated from typical travel times
                n_samp = X.shape[1]
                sr = 100
                p_sec = 0.0
                s_sec = min(10.0, (n_samp / sr) * 0.25)  # ~25% into window
                surface_sec = min(28.0, (n_samp / sr) * 0.55)  # ~55% into window
                rec = {
                    "waveform": X[idx].tolist(),
                    "label": "Earthquake",
                    "magnitude": float(row["magnitude"]),
                    "station": str(row.get("station", "")),
                    "network": str(row.get("network", "AK")),
                    "channel": str(row.get("channel", "")),
                    "event_time": str(row.get("event_time", "")),
                    "event_lat": float(row.get("event_lat", 0)),
                    "event_lon": float(row.get("event_lon", 0)),
                    "distance_km": float(row.get("distance_km", 0)),
                    "p_arrival_sec": p_sec,
                    "s_arrival_sec": s_sec,
                    "surface_arrival_sec": surface_sec,
                    "index": int(idx),
                }
                out["earthquake"][name].append(rec)
    else:
        # No metadata: pick random earthquakes and assign fake magnitude bins
        n_eq = min(6, len(eq_idx))
        chosen = np.random.default_rng(42).choice(eq_idx, size=n_eq, replace=False)
        mags = [2.5, 2.8, 3.2, 3.5, 4.0, 4.5][:n_eq]
        for idx, mag in zip(chosen, mags):
            name = "small" if mag < 3 else "medium" if mag < 4 else "large"
            n_samp = X.shape[1]
            sr = 100
            out["earthquake"][name].append({
                "waveform": X[idx].tolist(),
                "label": "Earthquake",
                "magnitude": mag,
                "p_arrival_sec": 0.0,
                "s_arrival_sec": min(10.0, (n_samp / sr) * 0.25),
                "surface_arrival_sec": min(28.0, (n_samp / sr) * 0.55),
                "index": int(idx),
            })

    # Single event: one well-documented earthquake with 5 stations for map view
    single_event = None
    if metadata is not None and "event_id" in metadata.columns:
        meta_se = metadata.copy()
        meta_se["row_index"] = np.arange(len(meta_se))
        eq_meta = meta_se[meta_se["label"] == 2]
        by_event = eq_meta.groupby("event_id")
        for event_id, grp in by_event:
            if len(grp) >= 5:
                rows = grp.head(5)
                first = rows.iloc[0]
                elat = float(first["event_lat"])
                elon = float(first["event_lon"])
                import math
                stations = []
                for i, (_, row) in enumerate(rows.iterrows()):
                    idx = int(row["row_index"])
                    dist_km = float(row["distance_km"])
                    az_deg = 360.0 * i / len(rows)
                    az_rad = math.radians(az_deg)
                    km_per_deg_lat = 111.32
                    km_per_deg_lon = 111.32 * math.cos(math.radians(elat))
                    slat = elat + (dist_km / km_per_deg_lat) * math.cos(az_rad)
                    slon = elon + (dist_km / km_per_deg_lon) * math.sin(az_rad)
                    stations.append({
                        "station": str(row.get("station", "")),
                        "network": str(row.get("network", "AK")),
                        "channel": str(row.get("channel", "")),
                        "distance_km": dist_km,
                        "lat": round(slat, 5),
                        "lon": round(slon, 5),
                        "waveform": X[idx].tolist(),
                    })
                single_event = {
                    "event": {
                        "event_id": str(event_id),
                        "event_time": str(first.get("event_time", "")),
                        "magnitude": float(first["magnitude"]),
                        "lat": elat,
                        "lon": elon,
                        "depth_km": float(first.get("event_depth_km", 0)),
                    },
                    "stations": stations,
                }
                break
    if single_event:
        out["single_event"] = single_event

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("Wrote", OUTPUT)

if __name__ == "__main__":
    main()
