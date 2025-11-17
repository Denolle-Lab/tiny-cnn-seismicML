"""
Training script for seismic signal classification.

This script trains the lightweight CNN on seismic data for signal classification.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.models import get_model
from src.data import SeismicDataset, DataAugmentation
from src.utils import Trainer, get_optimizer, get_scheduler


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_data(num_samples=1000, num_channels=3, seq_length=6000, num_classes=3):
    """
    Create dummy data for testing/demonstration.
    
    Replace this with actual data loading from files.
    """
    waveforms = np.random.randn(num_samples, num_channels, seq_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    return waveforms, labels


def main(args):
    """Main training function."""
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'model': {
                'type': 'standard',
                'num_classes': 3,
                'input_channels': 3,
                'input_length': 6000,
                'dropout_rate': 0.3
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'optimizer': 'adam',
                'scheduler': 'step',
                'early_stopping_patience': 10
            },
            'data': {
                'val_split': 0.2,
                'use_augmentation': True
            }
        }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create or load data
    # TODO: Replace with actual data loading
    print('Loading data...')
    waveforms, labels = create_dummy_data(
        num_samples=config.get('num_samples', 1000),
        num_channels=config['model']['input_channels'],
        seq_length=config['model']['input_length'],
        num_classes=config['model']['num_classes']
    )
    
    # Create augmentation if enabled
    transform = None
    if config['data'].get('use_augmentation', False):
        transform = DataAugmentation()
    
    # Create dataset
    dataset = SeismicDataset(waveforms, labels, transform=transform)
    
    # Split into train and validation
    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Create model
    print('Creating model...')
    model = get_model(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels'],
        input_length=config['model']['input_length'],
        dropout_rate=config['model'].get('dropout_rate', 0.3)
    )
    
    model = model.to(device)
    print(f'Model parameters: {model.count_parameters():,}')
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model,
        optimizer_type=config['training']['optimizer'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler if specified
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=config['training']['scheduler']
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train
    print('Starting training...')
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience')
    )
    
    print('\nTraining completed!')
    print(f'Best validation accuracy: {max(history["val_accuracies"]):.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train seismic CNN classifier')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    main(args)
