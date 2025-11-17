"""
Complete workflow test demonstrating all components.

This script tests:
1. Model creation and inspection
2. Data preprocessing
3. Dataset creation
4. Training (minimal epochs)
5. Model saving
6. Model loading and inference
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models import SeismicCNN, CompactSeismicCNN
from src.data import SeismicDataset, preprocess_seismogram, DataAugmentation
from src.utils import Trainer, get_optimizer


def test_complete_workflow():
    """Test complete workflow from data to prediction."""
    
    print("=" * 70)
    print("COMPLETE WORKFLOW TEST")
    print("=" * 70)
    
    # Configuration
    num_samples = 100
    num_classes = 3
    input_channels = 3
    input_length = 6000
    batch_size = 16
    num_epochs = 2
    
    # Step 1: Create dummy data
    print("\n1. Creating dummy seismic data...")
    waveforms = np.random.randn(num_samples, input_channels, input_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    print(f"   Created {num_samples} samples with shape {waveforms.shape}")
    
    # Step 2: Test preprocessing
    print("\n2. Testing preprocessing pipeline...")
    sample = waveforms[0]
    processed = preprocess_seismogram(sample, fs=100.0, target_length=input_length)
    print(f"   Preprocessed shape: {processed.shape}")
    print(f"   Mean: {np.mean(processed):.4f}, Std: {np.std(processed):.4f}")
    
    # Step 3: Create dataset
    print("\n3. Creating PyTorch dataset...")
    dataset = SeismicDataset(waveforms, labels)
    print(f"   Dataset size: {len(dataset)}")
    
    # Step 4: Create data loaders
    print("\n4. Creating data loaders...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Step 5: Create models
    print("\n5. Creating models...")
    model_standard = SeismicCNN(num_classes=num_classes, input_channels=input_channels)
    model_compact = CompactSeismicCNN(num_classes=num_classes, input_channels=input_channels)
    print(f"   Standard model parameters: {model_standard.count_parameters():,}")
    print(f"   Compact model parameters: {model_compact.count_parameters():,}")
    
    # Step 6: Test forward pass
    print("\n6. Testing forward pass...")
    device = torch.device('cpu')
    model_standard = model_standard.to(device)
    sample_batch, _ = next(iter(train_loader))
    with torch.no_grad():
        output = model_standard(sample_batch.to(device))
    print(f"   Input shape: {sample_batch.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Step 7: Train model
    print(f"\n7. Training model for {num_epochs} epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model_standard, optimizer_type='adam', lr=0.001)
    
    trainer = Trainer(
        model=model_standard,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir='/tmp/workflow_test'
    )
    
    history = trainer.train(num_epochs=num_epochs)
    print(f"   Training completed!")
    print(f"   Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"   Final val accuracy: {history['val_accuracies'][-1]:.2f}%")
    
    # Step 8: Load best model
    print("\n8. Loading best model...")
    checkpoint = torch.load('/tmp/workflow_test/best_model.pth')
    model_loaded = SeismicCNN(num_classes=num_classes, input_channels=input_channels)
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.eval()
    print(f"   Model loaded successfully!")
    
    # Step 9: Make predictions
    print("\n9. Making predictions on test data...")
    test_samples = waveforms[:5]
    test_tensor = torch.FloatTensor(test_samples)
    
    with torch.no_grad():
        predictions = model_loaded(test_tensor)
        probs = torch.softmax(predictions, dim=1)
        pred_classes = torch.argmax(probs, dim=1)
    
    class_names = ['Background', 'Urban', 'Tectonic']
    print(f"   Predictions:")
    for i, (pred_class, prob) in enumerate(zip(pred_classes, probs)):
        confidence = prob[pred_class].item()
        print(f"     Sample {i+1}: {class_names[pred_class]} (confidence: {confidence:.4f})")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nWorkflow Summary:")
    print("  ✓ Data generation and preprocessing")
    print("  ✓ Dataset and DataLoader creation")
    print("  ✓ Model creation (Standard and Compact)")
    print("  ✓ Training loop with validation")
    print("  ✓ Model checkpoint saving")
    print("  ✓ Model loading from checkpoint")
    print("  ✓ Inference and prediction")
    print("\nThe lightweight CNN framework is ready for seismic signal classification!")
    print("=" * 70)


if __name__ == '__main__':
    test_complete_workflow()
