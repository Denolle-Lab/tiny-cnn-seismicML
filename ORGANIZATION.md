# Project Organization Guide

This document provides an overview of the project structure for software engineers and data scientists.

## Directory Structure

```
tiny-cnn-seismicML/
│
├── notebooks/                  # Jupyter notebooks organized by workflow stage
│   ├── 01_data_exploration/   # EDA and data understanding
│   ├── 02_labeling/           # Feature-based labeling for supervised learning
│   ├── 03_training/           # CNN model training
│   └── 04_inference/          # Model deployment on continuous data
│
├── src/                       # Source code modules
│   ├── models/               # CNN model architectures
│   ├── data/                 # Data preprocessing utilities
│   └── utils/                # Training and utility functions
│
├── configs/                   # Configuration files
├── models/                    # Saved trained models (generated)
├── examples/                  # Basic usage examples (scripts)
│
├── train.py                   # CLI training script
├── predict.py                 # CLI inference script
└── requirements.txt           # Python dependencies
```

## Workflow Overview

### Standard ML Pipeline

This project follows a standard machine learning workflow:

```
Data Exploration → Labeling → Training → Inference
      (EDA)      (Supervised)  (CNN)    (Deployment)
```

### 1. Data Exploration (`notebooks/01_data_exploration/`)

**Purpose**: Understand the seismic data characteristics

**Activities**:
- Download data from Raspberry Shake network
- Explore station locations and availability
- Analyze signal types (earthquakes, traffic, noise)
- Assess data quality

**Key Notebooks**:
- `get_am_data.ipynb` - Data acquisition
- `anchorage_raspberry_shake_data.ipynb` - Station exploration
- `car_signal_detection.ipynb` - Signal analysis

**Output**: Understanding of data characteristics

---

### 2. Data Labeling (`notebooks/02_labeling/`)

**Purpose**: Create labeled training datasets

**Activities**:
- Extract seismic features (STA/LTA, kurtosis, spectral energy)
- Apply rule-based classification
- Generate labeled windows (5-second segments)
- Save training data

**Key Notebook**:
- `multi_class_labeling.ipynb` - Main labeling workflow

**Output**: 
- `labeled_data/windowed_waveforms_*.npy` - Waveform arrays
- `labeled_data/labels_*.npy` - Class labels (0=Noise, 1=Traffic, 2=Earthquake)
- `labeled_data/metadata_*.csv` - Feature metadata

---

### 3. Model Training (`notebooks/03_training/`)

**Purpose**: Train CNN classifier on labeled data

**Activities**:
- Load labeled datasets
- Split into train/validation/test (70/15/15)
- Train CompactSeismicCNN or SeismicCNN
- Evaluate with metrics and confusion matrix
- Save trained model

**Key Notebook**:
- `train_cnn_multiclass.ipynb` - Complete training pipeline

**Configuration**:
- Model: CompactSeismicCNN (lightweight) or SeismicCNN (standard)
- Loss: CrossEntropyLoss with class weights
- Optimizer: Adam (lr=0.001)
- Scheduler: StepLR
- Epochs: 50
- Batch size: 32

**Output**:
- `../../models/seismic_cnn_YYYYMMDD_HHMMSS.pth` - Trained model

---

### 4. Inference (`notebooks/04_inference/`)

**Purpose**: Apply trained model to continuous seismic data

**Activities**:
- Load trained model
- Download data from any station/time window
- Run sliding window predictions
- Visualize classification timeline
- Export results

**Key Notebook**:
- `predict_on_new_station.ipynb` - Main inference workflow

**Configuration**:
```python
NETWORK = "AM"                      # Network code
STATION = "RB38A"                   # Station code
START_TIME = "2024-11-27T17:00:00" # UTC start
END_TIME = "2024-11-27T17:30:00"   # UTC end
CONFIDENCE_THRESHOLD = 0.5         # Detection threshold
```

**Output**:
- Visualizations (timeline, probabilities, example waveforms)
- `predictions/*.csv` - Detailed prediction results

---

## For Software Engineers

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline**:
   ```bash
   # Step 1: Label data
   jupyter notebook notebooks/02_labeling/multi_class_labeling.ipynb
   
   # Step 2: Train model
   jupyter notebook notebooks/03_training/train_cnn_multiclass.ipynb
   
   # Step 3: Run inference
   jupyter notebook notebooks/04_inference/predict_on_new_station.ipynb
   ```

### Key Design Patterns

- **Modular Architecture**: Source code in `src/`, notebooks for workflow
- **Configuration Management**: YAML configs in `configs/`
- **Separation of Concerns**: Data, models, and utilities are separate modules
- **Reproducibility**: Fixed random seeds, versioned outputs with timestamps

### Data Flow

```
Raw Data (FDSN) 
    ↓
Feature Extraction (02_labeling)
    ↓
Labeled Datasets (.npy files)
    ↓
Training (03_training)
    ↓
Trained Model (.pth file)
    ↓
Inference (04_inference)
    ↓
Predictions (.csv file)
```

### Model Architecture

- **Input**: 1-channel seismogram (Z component), 500 samples (5 seconds @ 100 Hz)
- **Architecture**: 1D CNN with conv blocks, batch norm, pooling
- **Output**: 3 class probabilities (Noise, Traffic, Earthquake)
- **Variants**: 
  - CompactSeismicCNN: ~20K parameters (lightweight)
  - SeismicCNN: ~100K parameters (standard)

### API Usage

For programmatic usage, see `examples/usage_example.py`:

```python
from src.models.cnn import CompactSeismicCNN
import torch

# Load model
model = CompactSeismicCNN(num_classes=3, input_channels=1, input_length=500)
checkpoint = torch.load('models/seismic_cnn_latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
output = model(waveform_tensor)
probabilities = torch.softmax(output, dim=1)
```

## Best Practices

1. **Version Control**: All notebooks include timestamps in output filenames
2. **Documentation**: Each directory has its own README
3. **Reproducibility**: Fixed random seeds, saved configurations
4. **Modularity**: Reusable code in `src/`, experimentation in notebooks
5. **Data Management**: Generated data stored with source notebooks

## Common Tasks

### Training a New Model
```bash
cd notebooks/03_training
jupyter notebook train_cnn_multiclass.ipynb
# Or: python ../../train.py --config ../../configs/compact_config.yaml
```

### Running Inference on New Data
```bash
cd notebooks/04_inference
jupyter notebook predict_on_new_station.ipynb
# Configure station and time window in notebook
```

### Creating New Labeled Data
```bash
cd notebooks/02_labeling
jupyter notebook multi_class_labeling.ipynb
# Configure stations and time ranges in notebook
```

## Troubleshooting

### No labeled data found
- Run `notebooks/02_labeling/multi_class_labeling.ipynb` first
- Check that `labeled_data/` directory exists with .npy files

### No trained model found
- Run `notebooks/03_training/train_cnn_multiclass.ipynb` first
- Check that `models/` directory contains .pth files

### Import errors
- Ensure notebooks are run from their respective directories
- Notebooks use relative paths: `sys.path.insert(0, os.path.abspath('../..'))`

## Additional Resources

- Main README: `../README.md`
- Notebook workflows: `notebooks/README.md`
- Individual stage READMEs in each notebook subdirectory
- Configuration examples: `configs/`
