"""
Data preprocessing utilities for seismic signals.

This module provides functions for loading, preprocessing, and augmenting
seismogram data for training and inference. It does not import torch;
the PyTorch ``SeismicDataset`` lives in ``dataset.py``.
"""

import numpy as np
from scipy import signal


def normalize_waveform(waveform, method='standard'):
    """
    Normalize seismic waveform.
    
    Args:
        waveform (np.ndarray): Input waveform, shape (C, L) or (L,)
        method (str): Normalization method ('standard', 'minmax', or 'peak')
        
    Returns:
        np.ndarray: Normalized waveform
    """
    if method == 'standard':
        # Standardize to zero mean and unit variance
        mean = np.mean(waveform, axis=-1, keepdims=True)
        std = np.std(waveform, axis=-1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (waveform - mean) / std
    
    elif method == 'minmax':
        # Scale to [0, 1]
        wmin = np.min(waveform, axis=-1, keepdims=True)
        wmax = np.max(waveform, axis=-1, keepdims=True)
        wrange = wmax - wmin
        wrange = np.where(wrange == 0, 1, wrange)
        return (waveform - wmin) / wrange
    
    elif method == 'peak':
        # Normalize by peak amplitude
        peak = np.max(np.abs(waveform), axis=-1, keepdims=True)
        peak = np.where(peak == 0, 1, peak)
        return waveform / peak
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def bandpass_filter(waveform, lowcut, highcut, fs, order=4):
    """
    Apply bandpass filter to waveform.
    
    Args:
        waveform (np.ndarray): Input waveform
        lowcut (float): Low frequency cutoff (Hz)
        highcut (float): High frequency cutoff (Hz)
        fs (float): Sampling frequency (Hz)
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered waveform
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered = signal.sosfilt(sos, waveform, axis=-1)
    
    return filtered


def resample_waveform(waveform, original_rate, target_rate):
    """
    Resample waveform to target sampling rate.
    
    Args:
        waveform (np.ndarray): Input waveform
        original_rate (float): Original sampling rate (Hz)
        target_rate (float): Target sampling rate (Hz)
        
    Returns:
        np.ndarray: Resampled waveform
    """
    if original_rate == target_rate:
        return waveform
    
    num_samples = int(len(waveform) * target_rate / original_rate)
    resampled = signal.resample(waveform, num_samples, axis=-1)
    
    return resampled


def extract_window(waveform, window_length, offset=0):
    """
    Extract a fixed-length window from waveform.
    
    Args:
        waveform (np.ndarray): Input waveform
        window_length (int): Length of window to extract
        offset (int): Starting position (default: 0 for beginning)
        
    Returns:
        np.ndarray: Extracted window, zero-padded if necessary
    """
    waveform_length = waveform.shape[-1]
    
    if offset < 0:
        offset = 0
    
    # Create output array
    if waveform.ndim == 1:
        output = np.zeros(window_length)
        end = min(offset + window_length, waveform_length)
        output[:end-offset] = waveform[offset:end]
    else:
        output = np.zeros((waveform.shape[0], window_length))
        end = min(offset + window_length, waveform_length)
        output[:, :end-offset] = waveform[:, offset:end]
    
    return output


def preprocess_seismogram(waveform, fs=100.0, target_length=6000, 
                          lowcut=1.0, highcut=45.0, normalize_method='standard'):
    """
    Complete preprocessing pipeline for seismogram.
    
    Args:
        waveform (np.ndarray): Input waveform, shape (C, L) or (L,)
        fs (float): Sampling frequency (Hz)
        target_length (int): Target length for output
        lowcut (float): Bandpass low frequency (Hz)
        highcut (float): Bandpass high frequency (Hz)
        normalize_method (str): Normalization method
        
    Returns:
        np.ndarray: Preprocessed waveform
    """
    # Apply bandpass filter
    filtered = bandpass_filter(waveform, lowcut, highcut, fs)
    
    # Extract/pad to target length
    windowed = extract_window(filtered, target_length)
    
    # Normalize
    normalized = normalize_waveform(windowed, method=normalize_method)
    
    return normalized


class DataAugmentation:
    """
    Data augmentation for seismic signals.
    
    Args:
        noise_level (float): Standard deviation of Gaussian noise to add
        time_shift_range (int): Maximum time shift in samples
        amplitude_scale_range (tuple): Range for amplitude scaling (min, max)
    """
    
    def __init__(self, noise_level=0.01, time_shift_range=100, 
                 amplitude_scale_range=(0.8, 1.2)):
        self.noise_level = noise_level
        self.time_shift_range = time_shift_range
        self.amplitude_scale_range = amplitude_scale_range
    
    def add_noise(self, waveform):
        """Add Gaussian noise to waveform."""
        noise = np.random.normal(0, self.noise_level, waveform.shape)
        return waveform + noise
    
    def time_shift(self, waveform):
        """Apply random time shift to waveform."""
        shift = np.random.randint(-self.time_shift_range, self.time_shift_range)
        return np.roll(waveform, shift, axis=-1)
    
    def amplitude_scale(self, waveform):
        """Apply random amplitude scaling."""
        scale = np.random.uniform(*self.amplitude_scale_range)
        return waveform * scale
    
    def __call__(self, waveform):
        """Apply random augmentations."""
        waveform = self.add_noise(waveform)
        waveform = self.time_shift(waveform)
        waveform = self.amplitude_scale(waveform)
        return waveform
