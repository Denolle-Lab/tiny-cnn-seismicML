# tiny-cnn-seismicML

A lightweight PyTorch CNN for detecting and classifying seismic signals from Raspberry Shake seismograms.

## Overview

This repository provides compact convolutional neural networks designed specifically for seismic signal classification. The notebooks support both the newer AK Network Noise/Earthquake workflow and the earlier rule-based Noise/Traffic/Earthquake workflow for Raspberry Shake seismometer data.

### Features

- **Flexible Classification**: One on-disk label scheme (`src/data/labels.py`), per-model class subsets (earthquake-only today; Natural and Human models planned)
- **Lightweight Architecture**: Optimized for efficiency with minimal parameters
- **Two Model Variants**:
  - `SeismicCNN`: Standard model with good performance (~100K parameters)
  - `CompactSeismicCNN`: Ultra-compact model for edge devices (~20K parameters)
- **Complete Pipeline**: From data labeling to training to inference
- **AK Network Workflow**: Downloads professional Alaska Seismic Network windows for training
- **Browser Explainer**: React/TensorFlow.js app for explaining compact CNN predictions
- **Deployable Weights Pattern**: `models/<model-id>/metadata.json` + `weights.json` for browser consumers such as CLUE
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

1. **Label Data**: Download AK Noise/Earthquake windows or create rule-based labels
2. **Train Model**: Train the CNN on labeled data
3. **Predict**: Apply the trained model to new stations and time windows

### 1. Data Labeling

Generate the current AK Network Noise/Earthquake training data:

```bash
jupyter notebook notebooks/02_labeling/download_AK_only_data.ipynb
```

This notebook:
- Downloads earthquake and noise windows from the Alaska Seismic Network
- Uses 60-second, 100 Hz single-channel windows
- Saves `AK_waveforms_*`, `AK_labels_*`, and `AK_metadata_*` files to `notebooks/02_labeling/labeled_data/`

For the earlier three-class rule-based workflow, use:

```bash
jupyter notebook notebooks/02_labeling/multi_class_labeling.ipynb
```

That notebook:
- Downloads seismograms from multiple Raspberry Shake stations
- Extracts features (STA/LTA, kurtosis, spectral energy, etc.)
- Applies rule-based classification to label windows as Noise, Traffic, or Earthquake
- Saves labeled data to `notebooks/02_labeling/labeled_data/` directory

Note that this notebook cuts 5 s windows from 30 minutes around one earthquake, so its output does not mix with the 60 s AK windows.

For the anthropogenic classes (Traffic, Train, Aircraft; issues #10 to #12) use the continuous-window collector, which writes 60 s / 100 Hz windows in the AK file layout with a provisional label and a review sheet:

```bash
# one station-day of Raspberry Shake data, daytime = Traffic, 01-05 local = Noise
python scripts/collect_continuous_windows.py --network AM --station R4017 \
    --start 2026-09-09 --end 2026-09-10 --class-name Traffic \
    --label-from timeofday --review-sheet 24

# train passages or ADS-B landings from a CSV with a "time" column
python scripts/collect_continuous_windows.py --network AM --station R4017 \
    --start 2026-09-09 --end 2026-09-10 --class-name Train \
    --label-from events --events passages.csv
```

Output: `<NET>_<class>_waveforms_<stamp>.npy` (N, 6000), `_labels_` (global label integers), `_metadata_` (station, time, label method, per-window rms and band features, blank `reviewed` / `review_label` columns), a summary text file, and with `--review-sheet N` a PNG grid plus CSV to mark `keep` by eye. `--label-from rule` flags windows whose rms exceeds 3 times a quiet reference (on AM.R4017 daytime rms is 4.7 times the 02 to 04 local median; the 5 to 30 Hz band ratio does not separate them). Load the result in the training notebook with `DATA_SOURCE = 'AM_traffic'` or append it to the AK set with `EXTRA_SOURCES = ['AM_traffic']`.

Stations for these classes are listed in `configs/am_stations.yaml`: R1796 and R3130 at Romig Middle School (the partner school) and R4017 for contrast, matched to schools in `docs/am_station_school_matches.csv`. The config also fixes the collection days and label settings, so the whole AM pull is one command that regenerates the same station-days on any machine (file stamps differ; the notebook always picks the newest set per prefix):

```bash
git clone https://github.com/Denolle-Lab/tiny-cnn-seismicML.git && cd tiny-cnn-seismicML
python -m venv .venv && source .venv/bin/activate      # or a conda env
pip install -r requirements.txt
python scripts/collect_continuous_windows.py --config configs/am_stations.yaml
```

That writes `notebooks/02_labeling/labeled_data/AM_romig_traffic_<date>_*` for each day in the config (about 90 s and 46 MB per station-day; data files are gitignored, never committed). The AK earthquake/noise windows are **not** regenerated by this command: they come from `notebooks/02_labeling/download_AK_only_data.ipynb` (IRIS + ComCat downloads) and must be run once per machine, or copied from a machine that already has `AK_waveforms_*` files. The first traffic runs are one weekday and one Saturday per station, so in the notebook:

```python
DATA_SOURCE = 'ak'
EXTRA_SOURCES = ['AM_romig_traffic_2026-09-09', 'AM_romig_traffic_2026-09-12']
MODEL_CLASSES_KEY = 'rule_based'   # Noise / Traffic / Earthquake, or 'human_traffic' once train and aircraft exist
```

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
- Per-model class subset (`MODEL_CLASSES_KEY`): drops windows outside the subset and remaps labels to contiguous indices
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
WINDOW_LENGTH_SEC = 60.0     # Must match training
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
jupyter notebook notebooks/02_labeling/download_AK_only_data.ipynb
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
    num_classes=3,        # len(class_names) from select_classes, e.g. Noise, Earthquake, Avalanche
    input_channels=1,     # Single channel (Z component)
    input_length=6000,    # 60 seconds at 100 Hz
    dropout_rate=0.3
)

# Compact model for edge devices
model_compact = CompactSeismicCNN(
    num_classes=3,
    input_channels=1,
    input_length=6000
)
```

### Architecture Details

#### SeismicCNN (Standard)

- **Input**: 1-channel seismogram (Z component), length 6000 samples (60 seconds at 100 Hz)
- **Architecture**:
  - 4 convolutional blocks with batch normalization and max pooling
  - Global average pooling
  - 2 fully connected layers
  - Dropout for regularization
- **Output**: Class probabilities for the trained labels, usually Noise/Earthquake or Noise/Traffic/Earthquake
- **Parameters**: ~100,000

### CompactSeismicCNN (Lightweight)

- **Input**: Same as standard model
- **Architecture**:
  - 3 convolutional blocks (reduced filters)
  - Global average pooling
  - Single fully connected layer
- **Output**: Class probabilities for the trained labels
- **Parameters**: ~20,000

## Data Format

The model expects input data in the following format:

- **Shape**: `(batch_size, num_channels, sequence_length)`
- **Channels**: 1 (Z component of seismogram)
- **Sequence Length**: 6000 samples (60 seconds at 100 Hz, configurable)
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
window_length = 60.0  # seconds
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
├── docs/
│   ├── BROWSER_DEPLOYMENT.md
│   └── generating-model-weights.md
├── explainer-app/              # React + TensorFlow.js CNN explainer
├── notebooks/
│   ├── 01_data_exploration/    # Explore seismic data
│   ├── 02_labeling/            # Create labeled datasets
│   │   └── labeled_data/       # Generated labeled data (created during labeling)
│   ├── 03_training/            # Train CNN models
│   └── 04_inference/           # Deploy models on new data
│       └── predictions/        # Prediction results (created during inference)
├── models/
│   └── compact-v1/             # Example deployable TF.js model package
├── scripts/
│   ├── export_compact_weights_for_tfjs.py
│   ├── export_to_browser.py
│   └── export_waveforms_for_explainer.py
├── train.py                    # Command-line training script
├── predict.py                  # Command-line inference script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Notebook Organization

The `notebooks/` directory follows a standard ML workflow:

1. **`01_data_exploration/`** - Explore and understand seismic data
2. **`02_labeling/`** - Create labeled training datasets from AK downloads or rule-based features
3. **`03_training/`** - Train CNN models on labeled data
4. **`04_inference/`** - Apply trained models to continuous seismic data

Each directory contains its own README with detailed documentation. See `notebooks/README.md` for the complete workflow guide.

## Configuration

Training configuration can be customized in YAML files. Key parameters:

```yaml
model:
  type: 'standard'           # 'standard' or 'compact'
  classes: 'earthquake'      # key of src.data.MODEL_CLASSES or a list of class names
  input_channels: 1          # Vertical component
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
  waveforms: notebooks/02_labeling/labeled_data/AK_waveforms_<stamp>.npy  # optional
  labels: notebooks/02_labeling/labeled_data/AK_labels_<stamp>.npy        # optional
  val_split: 0.2
  use_augmentation: true
  sampling_rate: 100.0
  lowcut: 1.0
  highcut: 45.0
```

`num_classes` is derived from `classes`. Without `data.waveforms`/`data.labels`, `train.py` trains on dummy data.

## Classes

### Label scheme on disk

Every labeling notebook writes one global integer per window to `*_labels_*.npy`. The scheme lives in one place, `src/data/labels.py` (`LABEL_MAP`), and is append-only: 0, 1 and 2 are already on disk and never change.

| Label | Class | Status |
|---|---|---|
| 0 | Noise | on disk (AK and rule-based) |
| 1 | Traffic | on disk (rule-based Raspberry Shake), port tracked in [#12](https://github.com/Denolle-Lab/tiny-cnn-seismicML/issues/12) |
| 2 | Earthquake | on disk (AK, P-arrival windows) |
| 3 | Avalanche | collection tracked in [#9](https://github.com/Denolle-Lab/tiny-cnn-seismicML/issues/9) |
| 4 | Train | collection tracked in [#10](https://github.com/Denolle-Lab/tiny-cnn-seismicML/issues/10) |
| 5 | Aircraft | feasibility tracked in [#11](https://github.com/Denolle-Lab/tiny-cnn-seismicML/issues/11) |

### Per-model class subsets

A trained model separates a subset of these classes. `select_classes(X, y, key)` keeps only windows in the subset and remaps their labels to contiguous output indices 0..K-1, so `CrossEntropyLoss`, the checkpoint's `class_names`, and `models/<id>/metadata.json` all agree. Noise is always output index 0. Subsets are named in `MODEL_CLASSES`:

| Key | Output classes | Use |
|---|---|---|
| `earthquake` | Noise / Earthquake | deployed today (`models/compact-v2`, `models/standard-v1`) |
| `natural` | Noise / Earthquake / Avalanche | CLUE WaveRunner "Natural" model |
| `human` | Noise / Train / Aircraft | CLUE WaveRunner "Human" model |
| `human_traffic` | Noise / Traffic / Train / Aircraft | Human model with the rule-based traffic class |
| `rule_based` | Noise / Traffic / Earthquake | earlier three-class Raspberry Shake experiment |

Pick the subset with `MODEL_CLASSES_KEY` in `notebooks/03_training/train_cnn_multiclass.ipynb` or `model.classes` in a config YAML. The roadmap for the Natural and Human models is [#13](https://github.com/Denolle-Lab/tiny-cnn-seismicML/issues/13).

```python
from src.data import select_classes, LABEL_MAP

X, y, class_names = select_classes(X, y, 'natural')
# class_names == ['Noise', 'Earthquake', 'Avalanche']; y in {0, 1, 2}
```

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
