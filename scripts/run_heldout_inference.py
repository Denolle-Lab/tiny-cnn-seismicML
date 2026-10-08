#!/usr/bin/env python3
"""
Run every deployed model on the held-out inference subset
(datasets/heldout_inference/, from scripts/make_heldout_subset.py).

Windows are z-scored as in training. A window whose true class is not one of
the model's classes is still scored, and marked in_scope=False so it does not
count toward accuracy (e.g. an Aircraft window shown to a Noise/Earthquake model).

Usage (from repo root):
  python scripts/run_heldout_inference.py

Output: datasets/heldout_inference/heldout_predictions.csv  one row per model x window
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from src.models.cnn import CompactSeismicCNN, SeismicCNN
from plot_training_curves import MODELS_DIR, find_checkpoint

SUBSET_DIR = REPO_ROOT / "datasets" / "heldout_inference"
ARCHITECTURES = {"compact": CompactSeismicCNN, "standard": SeismicCNN}


def build_model(ckpt):
    model = ARCHITECTURES[ckpt["model_type"]](num_classes=ckpt["num_classes"],
                                              input_channels=ckpt["input_channels"],
                                              input_length=ckpt["input_length"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def main():
    X = np.load(SUBSET_DIR / "heldout_waveforms.npy")
    meta = pd.read_csv(SUBSET_DIR / "heldout_metadata.csv")
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-10)
    inputs = torch.from_numpy(X[:, np.newaxis, :])

    checkpoints = [(p, torch.load(p, map_location="cpu", weights_only=False))
                   for p in sorted(MODELS_DIR.glob("seismic_cnn_*.pth"))]

    rows = []
    for model_dir in sorted(d for d in MODELS_DIR.iterdir() if (d / "weights.json").exists()):
        ckpt_path, ckpt = find_checkpoint(model_dir, checkpoints)
        if ckpt is None:
            print(f"{model_dir.name}: no matching checkpoint, skipped")
            continue
        classes = ckpt["class_names"]
        with torch.no_grad():
            proba = torch.softmax(build_model(ckpt)(inputs), dim=1).numpy()
        for i, p in enumerate(proba):
            true = meta.label_name[i]
            rows.append({"model": model_dir.name, "checkpoint": ckpt_path.name,
                         "subset_id": meta.subset_id[i], "true": true,
                         "predicted": classes[int(p.argmax())], "confidence": round(float(p.max()), 4),
                         "in_scope": true in classes,
                         **{f"p_{c}": round(float(v), 4) for c, v in zip(classes, p)}})
        scored = [r for r in rows if r["model"] == model_dir.name and r["in_scope"]]
        n_ok = sum(r["true"] == r["predicted"] for r in scored)
        print(f"{model_dir.name}: {n_ok}/{len(scored)} in-scope windows correct")

    pd.DataFrame(rows).to_csv(SUBSET_DIR / "heldout_predictions.csv", index=False)


if __name__ == "__main__":
    main()
