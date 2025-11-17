"""
Tiny CNN for Seismic Signal Classification

A lightweight PyTorch CNN for detecting and classifying seismic signals
from Raspberry Shake seismograms.
"""

__version__ = '0.1.0'

from .models import SeismicCNN, CompactSeismicCNN, get_model
from .data import SeismicDataset, preprocess_seismogram, DataAugmentation
from .utils import Trainer, get_optimizer, get_scheduler

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
