# Model Inference & Deployment

This directory contains notebooks for deploying trained models on continuous seismic data.

## Notebooks

### `predict_on_new_station.ipynb`
**Main inference workflow** - Apply trained CNN to continuous data from any station.

**Purpose**: Run trained model on arbitrary time windows from any Raspberry Shake station
**Input**: Trained model from `../../models/`
**Output**: 
- Classification timeline visualizations
- Detection probability curves
- Prediction results saved to `predictions/*.csv`

**Features**:
- **Configurable**: Set any station, time window, and detection thresholds
- **Data Download**: Automatically fetches data from Raspberry Shake network
- **Sliding Window**: Processes continuous data with overlapping windows
- **Real-time Ready**: Can be adapted for real-time processing
- **Comprehensive Visualization**:
  - Raw waveform plot
  - Classification timeline (all predictions)
  - Earthquake detection probability over time
  - All class probabilities
  - Example waveforms from each detected class
- **Detection Summary**: 
  - Count and percentage of each class
  - High-confidence detections
  - Continuous segment identification
- **Export**: Save all predictions to CSV for further analysis

**Configuration Parameters**:
```python
NETWORK = "AM"                    # Network code
STATION = "RB38A"                 # Station code
START_TIME = "2024-11-27T17:00:00"  # UTC start time
END_TIME = "2024-11-27T17:30:00"    # UTC end time
CONFIDENCE_THRESHOLD = 0.5        # Detection threshold
```

## Workflow

1. Ensure a trained model exists in `../../models/`
2. Configure station and time window parameters
3. Run `predict_on_new_station.ipynb`
4. Review visualizations and detection summary
5. Optionally export results to CSV

## Output Directory

`predictions/` - Contains CSV files with detailed predictions:
- Window-by-window classifications
- Probability scores for all classes
- Timestamps for each detection
- Confidence levels

## Use Cases

- **Event Detection**: Find earthquakes in continuous data
- **Data Quality**: Identify noise vs. signal
- **Anthropogenic Analysis**: Detect traffic/urban signals
- **Real-time Monitoring**: Adapt for continuous station monitoring
- **Comparative Analysis**: Compare detection patterns across stations

## Next Steps

Use the exported CSV files for:
- Statistical analysis of detection patterns
- Performance evaluation against known events
- Integration with other monitoring systems
