# Data Exploration

This directory contains notebooks for exploring and understanding seismic data from Raspberry Shake stations.

## Notebooks

### `get_am_data.ipynb`
Download and explore seismic data from the Raspberry Shake network (AM network) in the Anchorage, Alaska region.

**Purpose**: Initial data acquisition and exploration
**Output**: Raw seismic data for analysis

### `anchorage_raspberry_shake_data.ipynb`
Comprehensive exploration of Raspberry Shake seismometer data from the Anchorage area.

**Purpose**: Understand station locations, data availability, and signal characteristics
**Output**: Visualizations and statistics about available data

### `car_signal_detection.ipynb`
Exploratory analysis of anthropogenic signals (traffic, cars) in seismic data.

**Purpose**: Understand characteristics of urban/traffic signals
**Output**: Feature analysis of anthropogenic signals

## Workflow

1. Start with `get_am_data.ipynb` to download data
2. Use `anchorage_raspberry_shake_data.ipynb` to explore station metadata and data quality
3. Use `car_signal_detection.ipynb` to understand different signal types

## Next Steps

After exploring the data, proceed to `02_labeling/` to create labeled training datasets.
