"""
Data package: labels, preprocessing, collection. Importing it does not
import torch, so the collection scripts run in an environment without it;
``SeismicDataset`` (PyTorch) is loaded on first access.
"""

from .preprocessing import (
    normalize_waveform,
    bandpass_filter,
    preprocess_seismogram,
    DataAugmentation
)
from .labels import (
    LABEL_MAP,
    NAME_TO_LABEL,
    MODEL_CLASSES,
    class_subset,
    classes_from_config,
    global_labels,
    infer_classes,
    select_classes,
)

__all__ = [
    'SeismicDataset',
    'normalize_waveform',
    'bandpass_filter',
    'preprocess_seismogram',
    'DataAugmentation',
    'LABEL_MAP',
    'NAME_TO_LABEL',
    'MODEL_CLASSES',
    'class_subset',
    'classes_from_config',
    'global_labels',
    'infer_classes',
    'select_classes',
]


def __getattr__(name):
    if name == 'SeismicDataset':
        from .dataset import SeismicDataset
        return SeismicDataset
    raise AttributeError(f"module 'src.data' has no attribute '{name}'")
