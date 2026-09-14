"""PyTorch Dataset wrapper for windowed seismograms (the only torch import in src.data)."""

import torch
from torch.utils.data import Dataset


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for seismic waveforms.
    
    Args:
        waveforms (np.ndarray): Array of waveforms, shape (N, C, L)
        labels (np.ndarray): Array of labels, shape (N,)
        transform (callable, optional): Optional transform to apply
    """
    
    def __init__(self, waveforms, labels, transform=None):
        self.waveforms = waveforms
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return torch.FloatTensor(waveform), torch.LongTensor([label]).squeeze()
