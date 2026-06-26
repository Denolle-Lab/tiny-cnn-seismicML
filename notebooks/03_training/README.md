# Model Training

This directory contains notebooks for training CNN models on labeled seismic data.

## Notebooks

### `train_cnn_multiclass.ipynb`
**Main training workflow** - Train a CNN classifier for two-class or three-class seismic signal classification.

**Purpose**: Train deep learning models to classify seismic signals
**Input**: Labeled data from `02_labeling/labeled_data/`
**Output**: Trained model saved to `../../models/seismic_cnn_*.pth`

**Training Configuration**:
- **Model**: CompactSeismicCNN (lightweight) or SeismicCNN (standard)
- **Classes**: 2 for AK Noise/Earthquake data, or 3 for rule-based Noise/Traffic/Earthquake data
- **Input**: Single-channel waveform windows, commonly 6000 samples (60 seconds at 100 Hz) for AK data
- **Data Split**: 70% train, 15% validation, 15% test
- **Loss**: CrossEntropyLoss with class weights (handles imbalanced data)
- **Optimizer**: Adam with model-specific learning rates
- **Scheduler**: ReduceLROnPlateau on validation loss
- **Epochs**: 60 maximum with early stopping
- **Batch Size**: 32

**Features**:
- Automatic train/val/test splitting
- Automatic class remapping when the dataset has labels 0 and 2 but no Traffic class
- Class-weighted loss for imbalanced datasets
- Learning rate scheduling
- Gradient clipping
- Training and validation metrics tracking
- Loss and accuracy curves
- Confusion matrix visualization
- Per-class performance metrics
- Model checkpointing (saves best model)

## Workflow

1. Ensure labeled data exists in `../02_labeling/labeled_data/`
2. Run `train_cnn_multiclass.ipynb` to train the model
3. Review training curves and test set performance
4. Trained model is automatically saved with timestamp

## Output

Trained models are saved to `../../models/` directory:
- Filename format: `seismic_cnn_<model>_<data-source>_YYYYMMDD_HHMMSS.pth`
- Includes model weights, configuration, and test accuracy

## Next Steps

After training, proceed to `04_inference/` to apply the model to new data.
