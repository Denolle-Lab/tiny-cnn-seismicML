"""
Lightweight CNN model for seismic signal detection and classification.

This module implements a compact convolutional neural network designed
for Raspberry Shake seismogram data classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicCNN(nn.Module):
    """
    Lightweight CNN for seismic signal classification.
    
    Architecture designed for 1D seismogram data with minimal parameters
    while maintaining good performance for signal detection and classification.
    
    Args:
        num_classes (int): Number of output classes (default: 3 for background, urban, tectonic)
        input_channels (int): Number of input channels (default: 3 for E-N-Z components)
        input_length (int): Length of input seismogram (default: 6000 samples)
        dropout_rate (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, num_classes=3, input_channels=3, input_length=6000, dropout_rate=0.3):
        super(SeismicCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_length = input_length
        
        # Convolutional blocks with batch normalization
        # Conv Block 1: Extract low-level features
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Block 2: Extract mid-level features
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Block 3: Extract high-level features
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Block 4: Deep features
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Global average pooling to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, input_length)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CompactSeismicCNN(nn.Module):
    """
    Ultra-lightweight CNN for resource-constrained environments.
    
    Even smaller model for deployment on edge devices like Raspberry Pi.
    
    Args:
        num_classes (int): Number of output classes (default: 3)
        input_channels (int): Number of input channels (default: 3)
        input_length (int): Length of input seismogram (default: 6000)
    """
    
    def __init__(self, num_classes=3, input_channels=3, input_length=6000):
        super(CompactSeismicCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Depthwise separable convolutions for efficiency
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simple classifier
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, input_length)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type='standard', **kwargs):
    """
    Factory function to create model instances.
    
    Args:
        model_type (str): Type of model ('standard' or 'compact')
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        nn.Module: Instantiated model
    """
    if model_type == 'standard':
        return SeismicCNN(**kwargs)
    elif model_type == 'compact':
        return CompactSeismicCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
