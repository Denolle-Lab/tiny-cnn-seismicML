#!/usr/bin/env python3
"""
Package a trained checkpoint as a CLUE-loadable model folder.

Usage (from repo root):
  python scripts/package_model.py --checkpoint models/seismic_cnn_standard_<...>.pth --model-id standard-v2
  python scripts/package_model.py --checkpoint <pth> --model-id compact-v3 --instrument-types H,L

Writes models/<model-id>/{metadata.json, weights.json, README.md}. Class
names, architecture, sampling rate and window length are read from the
checkpoint (written by scripts/train_multiclass.py). For checkpoints from
the notebook, pass --classes explicitly.

Instrument type is the SEED instrument code (second letter of the channel
code): H for both broadband BHZ/HHZ and Raspberry Shake EHZ, L for
low-gain. It is NOT the band code (the B in BHZ).
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.tfjs_export import (state_dict_to_tfjs, infer_architecture,  # noqa: E402
                                    num_classes_from_state)

SCHEMA = "https://collaborative-learning.concord.org/schemas/seismic-model/v1.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-id", required=True, help="e.g. standard-v2; also the folder name")
    ap.add_argument("--classes", default=None, help="comma-separated, only if the checkpoint lacks class_names")
    ap.add_argument("--instrument-types", default="H", help="comma-separated SEED instrument codes")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--force", action="store_true", help="overwrite an existing folder")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    arch = ckpt.get("architecture") if isinstance(ckpt, dict) else None
    arch = arch or infer_architecture(state)
    n_out = num_classes_from_state(state, arch)

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else \
        (ckpt.get("class_names") if isinstance(ckpt, dict) else None)
    if not classes:
        sys.exit("Checkpoint has no class_names; pass --classes Noise,Earthquake,...")
    if len(classes) != n_out:
        sys.exit(f"Checkpoint has {n_out} outputs but {len(classes)} class names were given")

    fs = float(ckpt.get("sampling_rate", 100)) if isinstance(ckpt, dict) else 100.0
    input_len = int(ckpt.get("input_length", 6000)) if isinstance(ckpt, dict) else 6000
    window_sec = input_len / fs

    out = args.out_root / args.model_id
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; bump the model id (CLUE keys events on it) or pass --force")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "weights.json", "w") as f:
        json.dump(state_dict_to_tfjs(state, arch), f, separators=(",", ":"))

    metadata = {
        "$schema": SCHEMA,
        "id": args.model_id,
        "architecture": arch,
        "class_names": classes,
        "sampling_rate": int(fs) if fs.is_integer() else fs,
        "window_duration": int(window_sec) if float(window_sec).is_integer() else window_sec,
        "instrument_types": [t.strip() for t in args.instrument_types.split(",")],
        "weightsUrl": "./weights.json",
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    lines = [f"# {args.model_id}", "",
             f"{arch} CNN, {len(classes)} classes: {', '.join(classes)}.", "",
             f"- Packaged {date.today()} from `{args.checkpoint.name}`",
             f"- Input: 1 channel, {input_len} samples ({window_sec:g} s at {fs:g} Hz), "
             f"bandpass {ckpt.get('window', {}).get('lowcut_hz', '?')}-{ckpt.get('window', {}).get('highcut_hz', '?')} Hz, "
             f"z-normalized per window" if isinstance(ckpt, dict) else "",
             f"- Parameters: {ckpt.get('parameters', '?'):,}" if isinstance(ckpt, dict) and ckpt.get("parameters") else "",
             f"- Training dataset: `{ckpt.get('dataset', '?')}`" if isinstance(ckpt, dict) else ""]
    if metrics:
        lines += ["", "## Test metrics (event-level split)", "",
                  f"- Accuracy: {metrics.get('test_accuracy', float('nan')) * 100:.1f}%",
                  "- Per-class recall: " + ", ".join(
                      f"{c} {r:.2f}" if r == r else f"{c} n/a" for c, r in zip(classes, metrics.get("test_recall", []))), ""]
        cm = metrics.get("confusion")
        if cm:
            lines += ["Confusion matrix (rows true, columns predicted):", "",
                      "| | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
            lines += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in zip(classes, cm)]
        if metrics.get("by_network"):
            lines += ["", "Per-network test accuracy: " + ", ".join(
                f"{n} {d['accuracy'] * 100:.1f}% (n={d['n']})" for n, d in metrics["by_network"].items())]
    lines += ["", "See `docs/generating-model-weights.md` for the format and "
              "`docs/multiclass-model-plan.md` for how the classes were built."]
    (out / "README.md").write_text("\n".join(l for l in lines if l is not None) + "\n")

    print(f"Wrote {out}/metadata.json, weights.json, README.md")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
