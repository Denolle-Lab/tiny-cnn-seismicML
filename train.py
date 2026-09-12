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
from src.data import SeismicDataset, DataAugmentation, class_subset, classes_from_config, global_labels, select_classes
from src.utils import Trainer, get_optimizer, get_scheduler


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_data(num_samples=1000, num_channels=3, seq_length=6000, classes='earthquake'):
    """
    Create dummy data for testing/demonstration.

    Labels are drawn from the global label integers of ``classes`` so the
    output goes through the same ``select_classes`` remap as real data.
    """
    waveforms = np.random.randn(num_samples, num_channels, seq_length).astype(np.float32)
    wanted = np.array(global_labels(classes))
    # One guaranteed window per class so select_classes never sees an empty class
    labels = np.concatenate([wanted, np.random.choice(wanted, max(num_samples - len(wanted), 0))])
    np.random.shuffle(labels)
    return waveforms, labels[:num_samples]


def load_labeled_arrays(waveforms_path, labels_path):
    """Load ``*_waveforms_*.npy`` / ``*_labels_*.npy`` written by notebooks/02_labeling."""
    try:
        waveforms = np.load(waveforms_path)
    except ValueError as err:
        if 'allow_pickle' not in str(err):
            raise
        # Object array: ragged windows pickled by the AK notebook
        raise ValueError(
            f'{waveforms_path} holds variable-length windows; crop them to a fixed '
            'length first (notebooks/03_training/train_cnn_multiclass.ipynb does this).'
        ) from err
    labels = np.load(labels_path)
    if waveforms.ndim == 2:  # (N, L) single channel -> (N, 1, L)
        waveforms = waveforms[:, np.newaxis, :]
    return waveforms.astype(np.float32), labels


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
                'classes': 'earthquake',
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
    
    # Set device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: CUDA ({torch.cuda.get_device_name(0)})')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device: MPS (Apple Silicon GPU)')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')
    
    # Resolve the class subset this model separates (see src/data/labels.py).
    # 'classes' is the source of truth; a legacy 'num_classes' is accepted.
    model_cfg = config['model']
    classes = classes_from_config(model_cfg)
    class_names = class_subset(classes)
    num_classes = len(class_names)
    print(f'Classes ({num_classes}): {class_names}')

    # Load labeled arrays if the config names them, else dummy data
    print('Loading data...')
    data_cfg = config['data']
    if data_cfg.get('waveforms') and data_cfg.get('labels'):
        waveforms, labels = load_labeled_arrays(data_cfg['waveforms'], data_cfg['labels'])
        print(f'Loaded {len(labels)} windows from {data_cfg["waveforms"]}')
    else:
        print('No data.waveforms/data.labels in config; using dummy data')
        waveforms, labels = create_dummy_data(
            num_samples=config.get('num_samples', 1000),
            num_channels=model_cfg['input_channels'],
            seq_length=model_cfg['input_length'],
            classes=classes
        )

    # Drop windows outside the subset and remap labels to 0..num_classes-1
    waveforms, labels, class_names = select_classes(waveforms, labels, classes)
    counts = np.bincount(labels, minlength=num_classes)
    print('Windows per class: ' + ', '.join(f'{n}={c}' for n, c in zip(class_names, counts)))
    
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
        model_type=model_cfg['type'],
        num_classes=num_classes,
        input_channels=model_cfg['input_channels'],
        input_length=model_cfg['input_length'],
        dropout_rate=model_cfg.get('dropout_rate', 0.3)
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
        save_dir=args.save_dir,
        metadata={
            'model_type': model_cfg['type'],
            'class_names': class_names,
            'num_classes': num_classes,
            'input_channels': model_cfg['input_channels'],
            'input_length': model_cfg['input_length'],
        }
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
