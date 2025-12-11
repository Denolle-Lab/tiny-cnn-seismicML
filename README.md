# tiny-cnn-seismicML

A lightweight PyTorch CNN for detecting and classifying seismic signals from Raspberry Shake seismograms.

## Overview

This repository provides a compact convolutional neural network designed specifically for seismic signal classification. The model can distinguish between different types of seismic signals including background noise, traffic/urban signals, and earthquake events from Raspberry Shake seismometer data.

### Features

- **Multi-Class Classification**: Distinguishes between Noise, Traffic, and Earthquake signals
- **Lightweight Architecture**: Optimized for efficiency with minimal parameters
- **Two Model Variants**:
  - `SeismicCNN`: Standard model with good performance (~100K parameters)
  - `CompactSeismicCNN`: Ultra-compact model for edge devices (~20K parameters)
- **Complete Pipeline**: From data labeling to training to inference
- **Preprocessing Pipeline**: Built-in utilities for seismogram preprocessing
- **Data Augmentation**: Support for training data augmentation
- **Easy to Use**: Interactive Jupyter notebooks and command-line scripts
- **Real-Time Capable**: Apply trained models to any station and time window

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- ObsPy >= 1.4.0 (for seismological data handling)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Denolle-Lab/tiny-cnn-seismicML.git
cd tiny-cnn-seismicML
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For Jupyter notebook support:
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=seismic-cnn
```

## Quick Start

### Workflow Overview

The complete workflow consists of three main steps:

1. **Label Data**: Use feature-based classification to create training labels
2. **Train Model**: Train the CNN on labeled data
3. **Predict**: Apply the trained model to new stations and time windows

### 1. Data Labeling

Generate labeled training data using the feature-based classifier:

```bash
jupyter notebook notebooks/02_labeling/multi_class_labeling.ipynb
```

This notebook:
- Downloads seismograms from multiple Raspberry Shake stations
- Extracts features (STA/LTA, kurtosis, spectral energy, etc.)
- Applies rule-based classification to label windows as Noise, Traffic, or Earthquake
- Saves labeled data to `notebooks/02_labeling/labeled_data/` directory

### 2. Model Training

Train the CNN using the labeled data:

```bash
jupyter notebook notebooks/03_training/train_cnn_multiclass.ipynb
```

Or use the command-line script:

```bash
python train.py --save-dir models
```

The training notebook provides:
- Train/validation/test split (70/15/15)
- Class-weighted loss for imbalanced data
- Learning rate scheduling
- Training and validation loss curves
- Confusion matrix and per-class metrics
- Model checkpointing (saves to `models/` directory)

### 3. Inference on New Stations

#### Using the Jupyter Notebook (Recommended)

The easiest way to apply the trained model to new seismic stations is using the prediction notebook:

```bash
jupyter notebook notebooks/04_inference/predict_on_new_station.ipynb
```

This notebook allows you to:
1. **Configure parameters**: Set the target station, time window, and detection thresholds
2. **Download data**: Automatically fetch data from Raspberry Shake network
3. **Make predictions**: Run the trained CNN on sliding windows
4. **Visualize results**: View classification timeline, probability curves, and example detections
5. **Export results**: Save predictions to CSV for further analysis

**Key configuration parameters:**

```python
# Station to analyze
NETWORK = "AM"
STATION = "RB38A"  # Any Raspberry Shake station
CHANNEL = "EHZ"

# Time window (use any arbitrary time period)
START_TIME = "2024-11-27T17:00:00"  # UTC
END_TIME = "2024-11-27T17:30:00"    # UTC (30 minutes)

# Detection settings
WINDOW_LENGTH_SEC = 5.0      # Must match training
WINDOW_OVERLAP = 0.5         # 50% overlap
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for detection
```

The notebook will:
- Download seismograms from the specified station and time window
- Preprocess the data (detrend, demean, taper)
- Split into overlapping windows
- Run predictions on each window
- Generate comprehensive visualizations showing:
  - Raw waveform
  - Classification timeline
  - Earthquake detection probability
  - All class probabilities over time
  - Example waveforms from each class
- Print detection summary with segment identification
- Optionally export results to CSV

#### Using the Command-Line Script

Alternatively, use the prediction script:

```bash
python predict.py --model-path checkpoints/best_model.pth --config configs/standard_config.yaml
```

### 4. Examples

Additional notebooks are organized by workflow stage in the `notebooks/` directory:

```bash
# Data exploration
jupyter notebook notebooks/01_data_exploration/get_am_data.ipynb

# Data labeling
jupyter notebook notebooks/02_labeling/multi_class_labeling.ipynb

# Model training
jupyter notebook notebooks/03_training/train_cnn_multiclass.ipynb

# Inference on new data
jupyter notebook notebooks/04_inference/predict_on_new_station.ipynb
```

See `notebooks/README.md` for detailed workflow documentation.

### SeismicCNN (Standard)

The repository includes two CNN architectures optimized for 1D seismic waveforms:

```python
from src.models import SeismicCNN, CompactSeismicCNN

# Standard model (default)
model = SeismicCNN(
    num_classes=3,        # Noise, Traffic, Earthquake
    input_channels=1,     # Single channel (Z component)
    input_length=500,     # 5 seconds at 100 Hz
    dropout_rate=0.3
)

# Compact model for edge devices
model_compact = CompactSeismicCNN(
    num_classes=3,
    input_channels=1,
    input_length=500
)
```

### Architecture Details

#### SeismicCNN (Standard)

- **Input**: 1-channel seismogram (Z component), length 500 samples (5 seconds at 100 Hz)
- **Architecture**:
  - 4 convolutional blocks with batch normalization and max pooling
  - Global average pooling
  - 2 fully connected layers
  - Dropout for regularization
- **Output**: 3 class probabilities (Noise, Traffic, Earthquake)
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
- **Channels**: 1 (Z component of seismogram)
- **Sequence Length**: 500 samples (5 seconds at 100 Hz, configurable)
- **Sampling Rate**: 100 Hz (default)
- **Window Overlap**: 50% overlap for sliding window predictions

### Preprocessing

The preprocessing pipeline includes:

1. Detrending (linear and demean)
2. Tapering (5% at edges)
3. Windowing to fixed length
4. Normalization (zero mean, unit variance per window)

```python
from obspy import read

# Read and preprocess seismogram
stream = read("seismogram.mseed")
stream.detrend('linear')
stream.detrend('demean')
stream.taper(max_percentage=0.05)

# Extract windows
window_length = 5.0  # seconds
overlap = 0.5        # 50%
# ... (see examples for complete windowing code)
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
├── notebooks/
│   ├── 01_data_exploration/    # Explore seismic data
│   ├── 02_labeling/            # Create labeled datasets
│   │   └── labeled_data/       # Generated labeled data (created during labeling)
│   ├── 03_training/            # Train CNN models
│   └── 04_inference/           # Deploy models on new data
│       └── predictions/        # Prediction results (created during inference)
├── models/                     # Saved trained models (created during training)
├── train.py                    # Command-line training script
├── predict.py                  # Command-line inference script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Notebook Organization

The `notebooks/` directory follows a standard ML workflow:

1. **`01_data_exploration/`** - Explore and understand seismic data
2. **`02_labeling/`** - Create labeled training datasets using feature-based classification
3. **`03_training/`** - Train CNN models on labeled data
4. **`04_inference/`** - Apply trained models to continuous seismic data

Each directory contains its own README with detailed documentation. See `notebooks/README.md` for the complete workflow guide.

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

1. **Noise** (Class 0): Ambient background noise
2. **Traffic** (Class 1): Anthropogenic/urban signals (e.g., traffic, human activity)
3. **Earthquake** (Class 2): Tectonic earthquake signals

### Classification Criteria

The model is trained on features including:
- **STA/LTA ratios**: Short-term to long-term amplitude ratios
- **Kurtosis**: Signal sharpness and impulsiveness
- **Spectral energy**: Energy distribution across frequency bands (0-5 Hz, 5-15 Hz, 15-30 Hz)
- **Dominant frequency**: Peak frequency content
- **Envelope characteristics**: Signal amplitude envelope properties

## Dependencies

See `requirements.txt` for a complete list of dependencies. Key packages include:

- **PyTorch** >= 2.0.0: Deep learning framework
- **NumPy** >= 1.24.0: Numerical computing
- **SciPy** >= 1.10.0: Scientific computing and signal processing
- **ObsPy** >= 1.4.0: Seismological data handling
- **scikit-learn** >= 1.3.0: Machine learning utilities
- **Matplotlib** >= 3.7.0: Visualization
- **Seaborn** >= 0.12.0: Statistical visualization
- **pandas** >= 2.0.0: Data manipulation

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
