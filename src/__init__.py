"""
Tiny CNN for Seismic Signal Classification

A lightweight PyTorch CNN for detecting and classifying seismic signals
from Raspberry Shake seismograms.

``src.data`` imports without torch; the model and trainer symbols below are
resolved on first access so data-collection scripts stay torch-free.
"""

__version__ = '0.1.0'

from .data import preprocess_seismogram, DataAugmentation

_LAZY = {
    'SeismicCNN': 'models', 'CompactSeismicCNN': 'models', 'get_model': 'models',
    'SeismicDataset': 'data',
    'Trainer': 'utils', 'get_optimizer': 'utils', 'get_scheduler': 'utils',
}

__all__ = [
    'SeismicCNN',
    'CompactSeismicCNN',
    'get_model',
    'SeismicDataset',
    'preprocess_seismogram',
    'DataAugmentation',
    'Trainer',
    'get_optimizer',
    'get_scheduler',
]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f'.{_LAZY[name]}', __name__), name)
    raise AttributeError(f"module 'src' has no attribute '{name}'")
