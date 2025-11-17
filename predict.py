"""
Inference script for seismic signal classification.

This script loads a trained model and performs predictions on seismic data.
"""

import argparse
import torch
import numpy as np
import yaml

from src.models import get_model
from src.data import preprocess_seismogram


class SeismicClassifier:
    """
    Classifier for seismic signals.
    
    Args:
        model_path (str): Path to trained model checkpoint
        config_path (str, optional): Path to configuration file
        device (str): Device to use ('cpu' or 'cuda')
    """
    
    def __init__(self, model_path, config_path=None, device='cpu'):
        self.device = torch.device(device)
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'model': {
                    'type': 'standard',
                    'num_classes': 3,
                    'input_channels': 3,
                    'input_length': 6000
                }
            }
        
        # Class names
        self.class_names = ['Background', 'Urban', 'Tectonic']
        
        # Load model
        self.model = get_model(
            model_type=self.config['model']['type'],
            num_classes=self.config['model']['num_classes'],
            input_channels=self.config['model']['input_channels'],
            input_length=self.config['model']['input_length']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f'Model loaded from {model_path}')
        print(f'Using device: {self.device}')
    
    def predict(self, waveform, fs=100.0, return_probs=False):
        """
        Predict class for a single waveform.
        
        Args:
            waveform (np.ndarray): Input waveform, shape (C, L) or (L,)
            fs (float): Sampling frequency (Hz)
            return_probs (bool): If True, return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Ensure correct shape
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        
        # Preprocess
        processed = preprocess_seismogram(
            waveform,
            fs=fs,
            target_length=self.config['model']['input_length']
        )
        
        # Add batch dimension
        input_tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        result = {
            'class': pred_class,
            'class_name': self.class_names[pred_class],
            'confidence': confidence
        }
        
        if return_probs:
            result['probabilities'] = {
                name: probs[0, i].item() 
                for i, name in enumerate(self.class_names)
            }
        
        return result
    
    def predict_batch(self, waveforms, fs=100.0):
        """
        Predict classes for multiple waveforms.
        
        Args:
            waveforms (np.ndarray): Input waveforms, shape (N, C, L)
            fs (float): Sampling frequency (Hz)
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for waveform in waveforms:
            result = self.predict(waveform, fs=fs, return_probs=True)
            results.append(result)
        
        return results


def main(args):
    """Main inference function."""
    
    # Create classifier
    classifier = SeismicClassifier(
        model_path=args.model_path,
        config_path=args.config,
        device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    
    # Example: Create dummy data for demonstration
    # Replace this with actual data loading
    print('\nGenerating example predictions...')
    
    num_samples = 5
    num_channels = classifier.config['model']['input_channels']
    seq_length = classifier.config['model']['input_length']
    
    # Create dummy waveforms
    waveforms = np.random.randn(num_samples, num_channels, seq_length).astype(np.float32)
    
    # Predict
    results = classifier.predict_batch(waveforms)
    
    # Display results
    print('\nPrediction Results:')
    print('-' * 70)
    for i, result in enumerate(results):
        print(f'\nSample {i+1}:')
        print(f"  Predicted Class: {result['class_name']} (class {result['class']})")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
    print('-' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seismic signal classification inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    main(args)
