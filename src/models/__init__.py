"""Model package initialization."""

from .cnn import SeismicCNN, CompactSeismicCNN, get_model

__all__ = ['SeismicCNN', 'CompactSeismicCNN', 'get_model']
