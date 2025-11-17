# tiny-cnn-seismicML

A lightweight PyTorch CNN for detecting and classifying seismic signals from Raspberry Shake seismograms.

## Overview

This repository provides a compact convolutional neural network designed specifically for seismic signal classification. The model can distinguish between different types of seismic signals including background noise, urban signals, and tectonic events.

### Features

- **Lightweight Architecture**: Optimized for efficiency with minimal parameters
- **Two Model Variants**:
  - `SeismicCNN`: Standard model with good performance (~100K parameters)
  - `CompactSeismicCNN`: Ultra-compact model for edge devices (~20K parameters)
- **Preprocessing Pipeline**: Built-in utilities for seismogram preprocessing
- **Data Augmentation**: Support for training data augmentation
- **Easy to Use**: Simple training and inference scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Denolle-Lab/tiny-cnn-seismicML.git
cd tiny-cnn-seismicML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Model Architecture

The repository includes two CNN architectures optimized for 1D seismic waveforms:

```python
from src.models import SeismicCNN, CompactSeismicCNN

# Standard model (default)
model = SeismicCNN(
    num_classes=3,        # Background, Urban, Tectonic
    input_channels=3,     # E-N-Z components
    input_length=6000,    # 60 seconds at 100 Hz
    dropout_rate=0.3
)

# Compact model for edge devices
model_compact = CompactSeismicCNN(
    num_classes=3,
    input_channels=3,
    input_length=6000
)
```

### 2. Training

Train the model using the provided training script:

```bash
# With default configuration
python train.py --save-dir checkpoints

# With custom configuration
python train.py --config configs/standard_config.yaml --save-dir checkpoints
```

Configuration files are available in the `configs/` directory:
- `standard_config.yaml`: Configuration for the standard model
- `compact_config.yaml`: Configuration for the compact model

### 3. Inference

Use a trained model for predictions:

```bash
python predict.py --model-path checkpoints/best_model.pth --config configs/standard_config.yaml
```

### 4. Examples

Run the example script to see model usage:

```bash
python examples/usage_example.py
```

## Model Architecture

### SeismicCNN (Standard)

- **Input**: 3-channel seismogram (E-N-Z components), length 6000 samples
- **Architecture**:
  - 4 convolutional blocks with batch normalization and max pooling
  - Global average pooling
  - 2 fully connected layers
  - Dropout for regularization
- **Output**: 3 class probabilities (Background, Urban, Tectonic)
- **Parameters**: ~100,000

### CompactSeismicCNN (Lightweight)

- **Input**: Same as standard model
- **Architecture**:
  - 3 convolutional blocks (reduced filters)
  - Global average pooling
  - Single fully connected layer
- **Output**: 3 class probabilities
- **Parameters**: ~20,000

## Data Format

The model expects input data in the following format:

- **Shape**: `(batch_size, num_channels, sequence_length)`
- **Channels**: 3 (E-N-Z components of seismogram)
- **Sequence Length**: 6000 samples (configurable)
- **Sampling Rate**: 100 Hz (default)

### Preprocessing

The preprocessing pipeline includes:

1. Bandpass filtering (1-45 Hz by default)
2. Windowing/padding to fixed length
3. Normalization (standard, minmax, or peak)

```python
from src.data import preprocess_seismogram

# Preprocess a seismogram
processed = preprocess_seismogram(
    waveform,
    fs=100.0,              # Sampling frequency
    target_length=6000,    # Target length
    lowcut=1.0,            # Bandpass low frequency
    highcut=45.0,          # Bandpass high frequency
    normalize_method='standard'
)
```

## Project Structure

```
tiny-cnn-seismicML/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py              # CNN model definitions
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py    # Data preprocessing utilities
│   └── utils/
│       ├── __init__.py
│       └── trainer.py          # Training utilities
├── configs/
│   ├── standard_config.yaml    # Standard model configuration
│   └── compact_config.yaml     # Compact model configuration
├── examples/
│   └── usage_example.py        # Usage examples
├── train.py                    # Training script
├── predict.py                  # Inference script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Configuration

Training configuration can be customized in YAML files. Key parameters:

```yaml
model:
  type: 'standard'           # 'standard' or 'compact'
  num_classes: 3
  input_channels: 3
  input_length: 6000
  dropout_rate: 0.3

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  optimizer: 'adam'          # 'adam', 'adamw', or 'sgd'
  scheduler: 'step'          # 'step', 'cosine', or 'plateau'
  early_stopping_patience: 10

data:
  val_split: 0.2
  use_augmentation: true
  sampling_rate: 100.0
  lowcut: 1.0
  highcut: 45.0
```

## Classes

The model classifies seismic signals into three categories:

1. **Background** (Class 0): Ambient noise
2. **Urban** (Class 1): Anthropogenic signals
3. **Tectonic** (Class 2): Earthquake signals

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- ObsPy >= 1.4.0 (for seismological data handling)
- PyYAML >= 6.0
- tqdm >= 4.65.0

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{tiny-cnn-seismicML,
  title={tiny-cnn-seismicML: Lightweight CNN for Seismic Signal Classification},
  author={Denolle Lab},
  year={2025},
  url={https://github.com/Denolle-Lab/tiny-cnn-seismicML}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and issues, please open an issue on GitHub.
