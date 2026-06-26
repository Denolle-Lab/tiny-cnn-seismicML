# Data Labeling

This directory contains notebooks for creating labeled training data.

## Notebooks

### `download_AK_only_data.ipynb`
**Current AK workflow** - Creates a two-class Noise/Earthquake dataset from Alaska Seismic Network data.

**Purpose**: Download professional seismic station windows for training the browser-compatible compact CNN
**Input**: AK event/station data downloaded by the notebook
**Output**:
- `labeled_data/AK_waveforms_*.npy` - 60-second, 100 Hz waveform windows
- `labeled_data/AK_labels_*.npy` - Class labels (0=Noise, 2=Earthquake before training remap)
- `labeled_data/AK_metadata_*.csv` - Event and station metadata for each window

### `multi_class_labeling.ipynb`
**Rule-based workflow** - Creates labeled training data for supervised learning.

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

1. Run `download_AK_only_data.ipynb` for the current AK Noise/Earthquake dataset
2. Or run `multi_class_labeling.ipynb` to create rule-based three-class datasets
3. Review the feature distributions and class balance
4. Labeled data is saved to `labeled_data/` directory

## Output Directory

`labeled_data/` - Contains all labeled datasets with timestamps:
- Windowed waveforms (numpy arrays)
- Labels (`AK_*`: 0=Noise, 2=Earthquake before training remap; rule-based: 0=Noise, 1=Traffic, 2=Earthquake)
- Metadata with extracted features

## Next Steps

After creating labeled data, proceed to `03_training/` to train the CNN model.
