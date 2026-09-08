"""Data package initialization.

``windows`` and ``catalog`` are importable without PyTorch; the
preprocessing helpers below need it.
"""

__all__ = []

try:
    from .preprocessing import (
        SeismicDataset,
        normalize_waveform,
        bandpass_filter,
        preprocess_seismogram,
        DataAugmentation
    )
    __all__ += [
        'SeismicDataset',
        'normalize_waveform',
        'bandpass_filter',
        'preprocess_seismogram',
        'DataAugmentation'
    ]
except ImportError:  # torch not installed
    pass
