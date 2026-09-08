"""
Tiny CNN for Seismic Signal Classification

A lightweight PyTorch CNN for detecting and classifying seismic signals
from Raspberry Shake seismograms.

The data-building side (src.data.windows, src.data.catalog) works without
PyTorch installed, so labeling can run on a machine that only has obspy.
"""

__version__ = '0.2.0'

__all__ = []

try:
    from .models import SeismicCNN, CompactSeismicCNN, get_model
    from .data import SeismicDataset, preprocess_seismogram, DataAugmentation
    from .utils import Trainer, get_optimizer, get_scheduler
    __all__ += [
        'SeismicCNN', 'CompactSeismicCNN', 'get_model',
        'SeismicDataset', 'preprocess_seismogram', 'DataAugmentation',
        'Trainer', 'get_optimizer', 'get_scheduler',
    ]
except ImportError:  # torch not installed: only the data tools are available
    pass
