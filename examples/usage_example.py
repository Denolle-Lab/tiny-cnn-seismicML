"""
Example script demonstrating model usage.

This script shows how to:
1. Create and inspect the model
2. Generate sample data
3. Run a forward pass
4. Save and load models
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.models import SeismicCNN, CompactSeismicCNN, get_model


def demo_model_creation():
    """Demonstrate model creation and inspection."""
    print("=" * 70)
    print("1. Model Creation and Architecture")
    print("=" * 70)
    
    # Create standard model
    model_standard = SeismicCNN(num_classes=3, input_channels=3, input_length=6000)
    print(f"\nStandard Model:")
    print(f"  Total parameters: {model_standard.count_parameters():,}")
    print(f"  Architecture: {model_standard}")
    
    # Create compact model
    model_compact = CompactSeismicCNN(num_classes=3, input_channels=3, input_length=6000)
    print(f"\nCompact Model:")
    print(f"  Total parameters: {model_compact.count_parameters():,}")
    print(f"  Architecture: {model_compact}")
    
    # Using factory function
    model = get_model('standard', num_classes=3, input_channels=3)
    print(f"\nModel from factory function:")
    print(f"  Type: standard")
    print(f"  Parameters: {model.count_parameters():,}")


def demo_forward_pass():
    """Demonstrate forward pass through the model."""
    print("\n" + "=" * 70)
    print("2. Forward Pass Example")
    print("=" * 70)
    
    # Create model
    model = SeismicCNN(num_classes=3, input_channels=3, input_length=6000)
    model.eval()
    
    # Create sample input
    batch_size = 4
    num_channels = 3
    seq_length = 6000
    
    sample_input = torch.randn(batch_size, num_channels, seq_length)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"  (batch_size, channels, sequence_length)")
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  (batch_size, num_classes)")
    
    # Get predictions
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    print(f"\nPredictions for batch:")
    class_names = ['Background', 'Urban', 'Tectonic']
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probs[i, pred_class].item()
        print(f"  Sample {i+1}: {class_names[pred_class]} (confidence: {confidence:.4f})")


def demo_save_load():
    """Demonstrate saving and loading models."""
    print("\n" + "=" * 70)
    print("3. Save and Load Model")
    print("=" * 70)
    
    # Create and save model
    model = SeismicCNN(num_classes=3, input_channels=3, input_length=6000)
    
    save_path = '/tmp/example_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Load model
    model_loaded = SeismicCNN(num_classes=3, input_channels=3, input_length=6000)
    model_loaded.load_state_dict(torch.load(save_path))
    model_loaded.eval()
    print(f"Model loaded successfully!")
    
    # Verify loaded model works
    sample_input = torch.randn(1, 3, 6000)
    with torch.no_grad():
        output = model_loaded(sample_input)
    print(f"Test forward pass output shape: {output.shape}")


def demo_preprocessing():
    """Demonstrate data preprocessing."""
    print("\n" + "=" * 70)
    print("4. Data Preprocessing Example")
    print("=" * 70)
    
    from src.data import preprocess_seismogram, normalize_waveform
    
    # Create sample waveform (3 channels, 8000 samples)
    waveform = np.random.randn(3, 8000).astype(np.float32)
    print(f"\nOriginal waveform shape: {waveform.shape}")
    
    # Preprocess
    preprocessed = preprocess_seismogram(
        waveform,
        fs=100.0,
        target_length=6000,
        lowcut=1.0,
        highcut=45.0,
        normalize_method='standard'
    )
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Mean: {np.mean(preprocessed, axis=1)}")
    print(f"Std: {np.std(preprocessed, axis=1)}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 70)
    print(" Seismic CNN Model - Usage Examples")
    print("*" * 70)
    
    demo_model_creation()
    demo_forward_pass()
    demo_save_load()
    demo_preprocessing()
    
    print("\n" + "*" * 70)
    print(" Examples completed successfully!")
    print("*" * 70)
    print("\n")


if __name__ == '__main__':
    main()
