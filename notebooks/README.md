# Notebooks

This directory contains Jupyter notebooks organized by workflow stage for the complete machine learning pipeline.

## Directory Structure

```
notebooks/
├── 01_data_exploration/    # Explore and understand seismic data
├── 02_labeling/            # Create labeled training datasets
├── 03_training/            # Train CNN models
└── 04_inference/           # Deploy models on new data
```

## Complete Workflow

### 1. Data Exploration (`01_data_exploration/`)
**Goal**: Understand your seismic data

- Download data from Raspberry Shake network
- Explore station locations and data availability
- Analyze different signal types (earthquakes, traffic, noise)
- Understand data quality and characteristics

**Key Notebook**: `get_am_data.ipynb`

---

### 2. Data Labeling (`02_labeling/`)
**Goal**: Create labeled training data for supervised learning

- Extract features from seismic waveforms (STA/LTA, kurtosis, spectral energy)
- Apply rule-based classification to label windows
- Generate labeled datasets with 3 classes: Noise, Traffic, Earthquake
- Save labeled data for model training

**Key Notebook**: `multi_class_labeling.ipynb`

**Output**: `labeled_data/` directory with windowed waveforms and labels

---

### 3. Model Training (`03_training/`)
**Goal**: Train a CNN classifier on labeled data

- Load labeled datasets
- Split data into train/validation/test sets
- Train CompactSeismicCNN or SeismicCNN model
- Evaluate performance with metrics and visualizations
- Save trained model for deployment

**Key Notebook**: `train_cnn_multiclass.ipynb`

**Output**: Trained model saved to `../models/seismic_cnn_*.pth`

---

### 4. Inference & Deployment (`04_inference/`)
**Goal**: Apply trained model to continuous seismic data

- Load trained model
- Download data from any station and time window
- Run sliding window predictions
- Visualize classification timeline and probabilities
- Export results for analysis

**Key Notebook**: `predict_on_new_station.ipynb`

**Output**: Predictions saved to `predictions/*.csv`

---

## Quick Start

For first-time users, follow this sequence:

```bash
# 1. Explore data
jupyter notebook 01_data_exploration/get_am_data.ipynb

# 2. Create labeled training data
jupyter notebook 02_labeling/multi_class_labeling.ipynb

# 3. Train the model
jupyter notebook 03_training/train_cnn_multiclass.ipynb

# 4. Run predictions on new data
jupyter notebook 04_inference/predict_on_new_station.ipynb
```

## For Software Engineers

This organization follows standard ML project structure:

1. **EDA (Exploratory Data Analysis)** → `01_data_exploration/`
2. **Data Preparation & Labeling** → `02_labeling/`
3. **Model Training** → `03_training/`
4. **Model Deployment/Inference** → `04_inference/`

Each directory contains:
- README.md explaining the workflow
- Notebooks with clear documentation
- Output directories for generated data/results

## Prerequisites

Ensure you have installed all dependencies:
```bash
pip install -r ../requirements.txt
```

For Jupyter support:
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=seismic-cnn
```

## Directory Details

Each subdirectory has its own README with:
- Detailed notebook descriptions
- Input/output specifications
- Configuration parameters
- Workflow instructions
- Next steps

See the README in each directory for more information.
