#!/usr/bin/env python3
"""
Plot training/validation loss and accuracy curves for deployed models.

Each models/<id>/ folder holds TF.js weights only, so the script finds the
PyTorch checkpoint it was exported from by matching the first conv kernel,
then plots the per-epoch history stored in that checkpoint.

Usage (from repo root):
  python scripts/plot_training_curves.py                      # every models/<id>/
  python scripts/plot_training_curves.py --models compact-v3 standard-v2

Output: docs/figures/training_curves_<id>.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def first_conv_kernel_tfjs(weights_path):
    """conv1/kernel from a TF.js weights.json, shape (kernel, in, out)."""
    return np.array(json.loads(weights_path.read_text())["conv1/kernel"])


def first_conv_kernel_torch(state_dict):
    """First Conv1d weight from a state_dict, transposed (out, in, k) -> (k, in, out)."""
    w = next(v for k, v in state_dict.items() if k.endswith("weight") and v.ndim == 3)
    return w.numpy().transpose(2, 1, 0)


def find_checkpoint(model_dir, checkpoints):
    """Return (path, checkpoint) whose first conv kernel matches the folder's weights.json."""
    kernel = first_conv_kernel_tfjs(model_dir / "weights.json")
    for path, ckpt in checkpoints:
        ck = first_conv_kernel_torch(ckpt["model_state_dict"])
        if ck.shape == kernel.shape and np.allclose(ck, kernel, atol=1e-6):
            return path, ckpt
    return None, None


def plot_history(model_id, ckpt_path, ckpt, out_path):
    h = ckpt["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    best = int(np.argmin(h["val_loss"])) + 1

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy (%)")]:
        ax.plot(epochs, h[f"train_{key}"], color="tab:blue", label="Train")
        ax.plot(epochs, h[f"val_{key}"], color="tab:orange", linestyle="--", label="Validation")
        ax.axvline(best, color="gray", linestyle=":", label=f"Lowest val loss (epoch {best})")
        ax.axvline(epochs[-1], color="black", linewidth=0.8, label=f"Saved weights (epoch {epochs[-1]})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)

    fig.suptitle(f"{model_id} · {' / '.join(ckpt['class_names'])} · "
                 f"test accuracy {ckpt['test_accuracy']:.4f} · {ckpt_path.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", help="model folder ids (default: all with weights.json)")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    args = parser.parse_args()

    model_dirs = sorted(d for d in MODELS_DIR.iterdir() if (d / "weights.json").exists())
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]

    checkpoints = [(p, torch.load(p, map_location="cpu", weights_only=False))
                   for p in sorted(MODELS_DIR.glob("seismic_cnn_*.pth"))]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in model_dirs:
        ckpt_path, ckpt = find_checkpoint(model_dir, checkpoints)
        if ckpt is None:
            print(f"{model_dir.name}: no matching checkpoint, skipped")
            continue
        out_path = args.out_dir / f"training_curves_{model_dir.name}.png"
        plot_history(model_dir.name, ckpt_path, ckpt, out_path)
        print(f"{model_dir.name}: {ckpt_path.name} -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
