"""Configuration for XR-world dual-wrist alignment.

The matrices below are in the raw Vuer/OpenXR world convention (metres).
Replace them with targets measured in the same convention, or pass --alignment-target-config.
"""
import json
from pathlib import Path
import numpy as np

DEFAULT_TARGETS = {
    "left": np.array([[1., 0., 0., -0.30], [0., 1., 0., 1.20], [0., 0., 1., -0.20], [0., 0., 0., 1.]]),
    "right": np.array([[1., 0., 0., 0.30], [0., 1., 0., 1.20], [0., 0., 1., -0.20], [0., 0., 0., 1.]]),
}

def load_targets(path=None):
    if not path:
        return {key: value.copy() for key, value in DEFAULT_TARGETS.items()}
    data = json.loads(Path(path).read_text())
    return {key: np.asarray(data[key], dtype=float).reshape(4, 4) for key in ("left", "right")}
