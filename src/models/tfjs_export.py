"""
Convert a trained checkpoint's state_dict to the TF.js weight JSON that
CLUE and the explainer app load. Shared by scripts/package_model.py and
the two standalone export scripts.

Layer names must match the build functions registered in CLUE's
architecture registry (``compact`` and ``standard``); weights are loaded
by name on the JS side.
"""

from __future__ import annotations

import numpy as np

ARCHITECTURES = {
    "compact": {"convs": ["conv1", "conv2", "conv3"], "bns": ["bn1", "bn2", "bn3"],
                "denses": ["fc"], "head": "fc"},
    "standard": {"convs": ["conv1", "conv2", "conv3", "conv4"], "bns": ["bn1", "bn2", "bn3", "bn4"],
                 "denses": ["fc1", "fc2"], "head": "fc2"},
}


def infer_architecture(state: dict) -> str:
    keys = set(state.keys())
    if "fc2.weight" in keys and "conv4.weight" in keys:
        return "standard"
    if "fc.weight" in keys and "conv3.weight" in keys and "conv4.weight" not in keys:
        return "compact"
    raise ValueError("state_dict does not match the compact or standard architecture")


def num_classes_from_state(state: dict, arch: str) -> int:
    return int(state[f"{ARCHITECTURES[arch]['head']}.weight"].shape[0])


def state_dict_to_tfjs(state: dict, arch: str) -> dict:
    spec = ARCHITECTURES[arch]
    to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    out = {"num_classes": num_classes_from_state(state, arch)}
    for name in spec["convs"]:
        w = to_np(state[f"{name}.weight"])            # (out, in, k)
        out[f"{name}/kernel"] = np.transpose(w, (2, 1, 0)).tolist()
        out[f"{name}/bias"] = to_np(state[f"{name}.bias"]).tolist()
    for name in spec["bns"]:
        out[f"{name}/gamma"] = to_np(state[f"{name}.weight"]).tolist()
        out[f"{name}/beta"] = to_np(state[f"{name}.bias"]).tolist()
        out[f"{name}/moving_mean"] = to_np(state[f"{name}.running_mean"]).tolist()
        out[f"{name}/moving_variance"] = to_np(state[f"{name}.running_var"]).tolist()
    for name in spec["denses"]:
        w = to_np(state[f"{name}.weight"])            # (out, in)
        out[f"{name}/kernel"] = np.transpose(w, (1, 0)).tolist()
        out[f"{name}/bias"] = to_np(state[f"{name}.bias"]).tolist()
    return out
