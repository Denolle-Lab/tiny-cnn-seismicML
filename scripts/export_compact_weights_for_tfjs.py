#!/usr/bin/env python3
"""
Export CompactSeismicCNN PyTorch weights to JSON for TensorFlow.js.

The TF.js model is built with the same architecture in the explainer-app;
this script saves state_dict tensors with correct transposes for conv/dense.

Usage (from repo root):
  python scripts/export_compact_weights_for_tfjs.py
  python scripts/export_compact_weights_for_tfjs.py --model_path models/seismic_cnn_compact_<...>.pth

Output: explainer-app/public/models/compact_weights.json
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
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)
CompactSeismicCNN = cnn.CompactSeismicCNN


def find_latest_compact_checkpoint() -> Optional[Path]:
    """Return the newest local compact checkpoint when --model_path is omitted."""
    candidates = sorted((REPO_ROOT / "models").glob("seismic_cnn_compact*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export_weights(model_path: Path, output_path: Path) -> None:
    """Load Compact model and export weights as JSON for TF.js."""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    # Infer num_classes from saved fc weight shape (out_features, 64)
    num_classes = int(state["fc.weight"].shape[0])
    model = CompactSeismicCNN(num_classes=num_classes, input_channels=1, input_length=6000)
    model.load_state_dict(state, strict=True)
    model.eval()

    out = {"num_classes": num_classes}

    # Conv1d: PyTorch (outCh, inCh, k) -> TF.js (k, inCh, outCh)
    for name in ["conv1", "conv2", "conv3"]:
        w = state[f"{name}.weight"].numpy()
        # (out, in, k) -> (k, in, out)
        out[f"{name}/kernel"] = np.transpose(w, (2, 1, 0)).tolist()
        out[f"{name}/bias"] = state[f"{name}.bias"].numpy().tolist()

    # BatchNorm: PyTorch weight=gamma, bias=beta, running_mean, running_var
    for name in ["bn1", "bn2", "bn3"]:
        out[f"{name}/gamma"] = state[f"{name}.weight"].numpy().tolist()
        out[f"{name}/beta"] = state[f"{name}.bias"].numpy().tolist()
        out[f"{name}/moving_mean"] = state[f"{name}.running_mean"].numpy().tolist()
        out[f"{name}/moving_variance"] = state[f"{name}.running_var"].numpy().tolist()

    # Dense: PyTorch Linear(64, 3) weight (3, 64) -> TF.js (64, 3)
    w = state["fc.weight"].numpy()
    out["fc/kernel"] = np.transpose(w, (1, 0)).tolist()
    out["fc/bias"] = state["fc.bias"].numpy().tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"Exported weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Compact CNN weights for TF.js")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to .pth checkpoint. Defaults to the newest models/seismic_cnn_compact*.pth file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "explainer-app" / "public" / "models",
        help="Directory to write compact_weights.json",
    )
    args = parser.parse_args()

    model_path = args.model_path or find_latest_compact_checkpoint()
    if model_path is None:
        print("No compact checkpoints found. Pass --model_path explicitly.")
        sys.exit(1)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    export_weights(model_path, args.output_dir / "compact_weights.json")


if __name__ == "__main__":
    main()
