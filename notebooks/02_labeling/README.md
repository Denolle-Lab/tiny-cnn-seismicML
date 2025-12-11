# Data Labeling

This directory contains notebooks for creating labeled training data using feature-based classification.

## Notebooks

### `multi_class_labeling.ipynb`
**Main labeling workflow** - Creates labeled training data for supervised learning.

**Purpose**: Generate labeled datasets by applying rule-based feature classification
**Input**: Raw seismic data from multiple stations
**Output**: 
- `labeled_data/windowed_waveforms_*.npy` - Windowed seismogram arrays
- `labeled_data/labels_*.npy` - Class labels (0=Noise, 1=Traffic, 2=Earthquake)
- `labeled_data/metadata_*.csv` - Feature metadata for each window

**Classification Strategy**:
- Extracts features: STA/LTA ratios, kurtosis, spectral energy, dominant frequency
- Applies rule-based classification to label windows
- Uses 5-second windows with 50% overlap
- Outputs normalized waveforms ready for CNN training

## Workflow

1. Run `multi_class_labeling.ipynb` to create labeled datasets
2. Review the feature distributions and class balance
3. Labeled data is saved to `labeled_data/` directory

## Output Directory

`labeled_data/` - Contains all labeled datasets with timestamps:
- Windowed waveforms (numpy arrays)
- Labels (0=Noise, 1=Traffic, 2=Earthquake)
- Metadata with extracted features

## Next Steps

After creating labeled data, proceed to `03_training/` to train the CNN model.
