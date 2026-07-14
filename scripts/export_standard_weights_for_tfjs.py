#!/usr/bin/env python3
"""
Export SeismicCNN (standard) PyTorch weights to JSON for TensorFlow.js.

Sibling of export_compact_weights_for_tfjs.py for the standard architecture
(4 conv blocks + 2 dense layers). Saves state_dict tensors with the correct
transposes for conv/dense layers; batchnorm params are renamed to the TF.js
gamma/beta/moving_mean/moving_variance convention.

Usage (from repo root):
  python scripts/export_standard_weights_for_tfjs.py
  python scripts/export_standard_weights_for_tfjs.py --model_path models/seismic_cnn_standard_<...>.pth

Output: explainer-app/public/models/standard_weights.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import only the cnn module to avoid pulling in scipy etc.
import importlib.util
spec = importlib.util.spec_from_file_location("cnn", REPO_ROOT / "src" / "models" / "cnn.py")
if spec is None or spec.loader is None:
    raise ImportError("Failed to load src/models/cnn.py (importlib spec/loader is missing).")
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)
SeismicCNN = cnn.SeismicCNN


def find_latest_standard_checkpoint() -> Optional[Path]:
    """Return the newest local standard checkpoint when --model_path is omitted."""
    candidates = sorted((REPO_ROOT / "models").glob("seismic_cnn_standard*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export_weights(model_path: Path, output_path: Path) -> None:
    """Load standard SeismicCNN model and export weights as JSON for TF.js."""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    # Infer shapes from the checkpoint so we don't rely on class defaults:
    #   fc2.weight is (num_classes, 64); conv1.weight is (out, in_channels, k).
    num_classes = int(state["fc2.weight"].shape[0])
    input_channels = int(state["conv1.weight"].shape[1])
    model = SeismicCNN(num_classes=num_classes, input_channels=input_channels, input_length=6000)
    model.load_state_dict(state, strict=True)
    model.eval()

    out = {"num_classes": num_classes}

    # Conv1d: PyTorch (outCh, inCh, k) -> TF.js (k, inCh, outCh)
    for name in ["conv1", "conv2", "conv3", "conv4"]:
        w = state[f"{name}.weight"].numpy()
        out[f"{name}/kernel"] = np.transpose(w, (2, 1, 0)).tolist()
        out[f"{name}/bias"] = state[f"{name}.bias"].numpy().tolist()

    # BatchNorm: PyTorch weight=gamma, bias=beta, running_mean, running_var
    for name in ["bn1", "bn2", "bn3", "bn4"]:
        out[f"{name}/gamma"] = state[f"{name}.weight"].numpy().tolist()
        out[f"{name}/beta"] = state[f"{name}.bias"].numpy().tolist()
        out[f"{name}/moving_mean"] = state[f"{name}.running_mean"].numpy().tolist()
        out[f"{name}/moving_variance"] = state[f"{name}.running_var"].numpy().tolist()

    # Dense: PyTorch Linear weight (out, in) -> TF.js (in, out)
    for name in ["fc1", "fc2"]:
        w = state[f"{name}.weight"].numpy()
        out[f"{name}/kernel"] = np.transpose(w, (1, 0)).tolist()
        out[f"{name}/bias"] = state[f"{name}.bias"].numpy().tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"Exported weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export standard SeismicCNN weights for TF.js")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to .pth checkpoint. Defaults to the newest models/seismic_cnn_standard*.pth file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "explainer-app" / "public" / "models",
        help="Directory to write standard_weights.json",
    )
    args = parser.parse_args()

    model_path = args.model_path or find_latest_standard_checkpoint()
    if model_path is None:
        print("No standard checkpoints found. Pass --model_path explicitly.")
        sys.exit(1)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    export_weights(model_path, args.output_dir / "standard_weights.json")


if __name__ == "__main__":
    main()
