#!/usr/bin/env python3
"""
Export PyTorch seismic CNN models to browser-compatible formats.

This script exports trained PyTorch models to ONNX format, which can then be
converted to TensorFlow.js for browser deployment. It handles both the compact
and standard model architectures.

Usage:
    python scripts/export_to_browser.py --model_path path/to/model.pth --output_dir browser_demo/models
    
Requirements:
    pip install torch onnx onnx-tf tensorflowjs
"""

import argparse
import json
import torch
import onnx
from pathlib import Path
import sys

# Add project root to path to import models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.cnn import SeismicCNN, CompactSeismicCNN

DEFAULT_CLASS_NAMES = ["Noise", "Traffic", "Earthquake"]
DEFAULT_SAMPLING_RATE = 100
DEFAULT_INPUT_LENGTH = 6000


def export_pytorch_to_onnx(model, output_path, model_name="seismic_cnn", input_channels=1, input_length=DEFAULT_INPUT_LENGTH):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model instance
        output_path: Path to save ONNX model
        model_name: Name for the model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input matching the expected shape (batch_size, channels, samples)
    dummy_input = torch.randn(1, input_channels, input_length)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,  # Compatible with TensorFlow.js
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Exported ONNX model to {output_path}")
    
    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model verification passed")


def get_state_dict(checkpoint):
    """Return the model state dict from either a raw state dict or notebook checkpoint."""
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


def infer_checkpoint_info(checkpoint, model_type):
    """Infer browser metadata from the notebook checkpoint when available."""
    state_dict = get_state_dict(checkpoint)
    num_classes = checkpoint.get('num_classes') if isinstance(checkpoint, dict) else None
    input_channels = checkpoint.get('input_channels') if isinstance(checkpoint, dict) else None
    input_length = checkpoint.get('input_length') if isinstance(checkpoint, dict) else None
    class_names = checkpoint.get('class_names') if isinstance(checkpoint, dict) else None

    if num_classes is None:
        fc_key = 'fc.weight' if model_type == 'compact' else 'fc2.weight'
        num_classes = int(state_dict[fc_key].shape[0])
    if input_channels is None:
        input_channels = int(state_dict['conv1.weight'].shape[1])
    if input_length is None:
        input_length = DEFAULT_INPUT_LENGTH
    if class_names is None:
        class_names = ["Noise", "Earthquake"] if num_classes == 2 else DEFAULT_CLASS_NAMES[:num_classes]

    return {
        "num_classes": int(num_classes),
        "input_channels": int(input_channels),
        "input_length": int(input_length),
        "class_names": list(class_names),
    }


def build_model(model_type, checkpoint_info):
    """Create the matching PyTorch model for export."""
    kwargs = {
        "num_classes": checkpoint_info["num_classes"],
        "input_channels": checkpoint_info["input_channels"],
        "input_length": checkpoint_info["input_length"],
    }
    if model_type == "compact":
        return CompactSeismicCNN(**kwargs)
    return SeismicCNN(**kwargs)


def create_model_metadata(model_path, output_dir, model, model_type, checkpoint_info):
    """
    Create metadata JSON file with model information.
    
    Args:
        model_path: Path to the PyTorch model file
        output_dir: Directory to save metadata
        model_type: Type of model ("compact" or "standard")
    """
    param_count = sum(p.numel() for p in model.parameters())
    
    # Extract timestamp from model filename if available
    model_filename = Path(model_path).stem
    
    metadata = {
        "model_name": model_filename,
        "model_type": model_type,
        "architecture": model.__class__.__name__,
        "num_parameters": param_count,
        "input_shape": [checkpoint_info["input_channels"], checkpoint_info["input_length"]],
        "output_classes": checkpoint_info["num_classes"],
        "class_names": checkpoint_info["class_names"],
        "sampling_rate": DEFAULT_SAMPLING_RATE,  # Hz
        "window_duration": checkpoint_info["input_length"] / DEFAULT_SAMPLING_RATE,
        "preprocessing": {
            "detrend": True,
            "bandpass_filter": {
                "freqmin": 1.0,
                "freqmax": 45.0,
                "corners": 4
            },
            "normalization": "per_channel"
        }
    }
    
    metadata_path = output_dir / f"{model_filename}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {metadata_path}")
    return metadata


def generate_conversion_script(onnx_path, output_dir):
    """
    Generate a bash script to convert ONNX to TensorFlow.js.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for TensorFlow.js model
    """
    script_path = output_dir / "convert_to_tfjs.sh"
    
    onnx_name = Path(onnx_path).stem
    tf_model_dir = output_dir / f"{onnx_name}_tf"
    tfjs_model_dir = output_dir / onnx_name
    
    script_content = f"""#!/bin/bash
# Auto-generated conversion script
# Converts ONNX model to TensorFlow.js format

echo "Step 1: Converting ONNX to TensorFlow SavedModel..."
onnx-tf convert -i {onnx_path} -o {tf_model_dir}

echo "Step 2: Converting TensorFlow SavedModel to TensorFlow.js..."
tensorflowjs_converter \\
    --input_format=tf_saved_model \\
    --output_format=tfjs_graph_model \\
    --signature_name=serving_default \\
    --saved_model_tags=serve \\
    {tf_model_dir} \\
    {tfjs_model_dir}

echo "✓ Conversion complete!"
echo "Model saved to: {tfjs_model_dir}"
echo ""
echo "To use in browser, copy the model directory to your web server:"
echo "  cp -r {tfjs_model_dir} /path/to/browser_demo/models/"
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    script_path.chmod(0o755)
    
    print(f"✓ Created conversion script: {script_path}")
    print(f"  Run: bash {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Export seismic CNN models to browser-compatible formats"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained PyTorch model (.pth file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="browser_demo/models",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["compact", "standard"],
        help="Type of model architecture"
    )
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="Export all models from trained_models directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.export_all:
        # Find all trained models
        trained_models_dir = project_root / "models"
        if not trained_models_dir.exists():
            print(f"Error: {trained_models_dir} does not exist")
            return
        
        model_files = list(trained_models_dir.glob("*.pth"))
        if not model_files:
            print(f"No .pth files found in {trained_models_dir}")
            return
        
        print(f"Found {len(model_files)} model(s) to export\n")
        
        for model_path in model_files:
            # Determine model type from filename
            if "compact" in model_path.stem.lower():
                model_type = "compact"
            else:
                model_type = "standard"
            
            print(f"\n{'='*60}")
            print(f"Processing: {model_path.name}")
            print(f"Model type: {model_type}")
            print(f"{'='*60}\n")
            
            # Load model weights
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                checkpoint_info = infer_checkpoint_info(checkpoint, model_type)
                model = build_model(model_type, checkpoint_info)
                model.load_state_dict(get_state_dict(checkpoint))
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue
            
            # Export to ONNX
            onnx_path = output_dir / f"{model_path.stem}.onnx"
            export_pytorch_to_onnx(
                model,
                onnx_path,
                model_path.stem,
                checkpoint_info["input_channels"],
                checkpoint_info["input_length"],
            )
            
            # Create metadata
            create_model_metadata(model_path, output_dir, model, model_type, checkpoint_info)
            
            # Generate conversion script
            generate_conversion_script(onnx_path, output_dir)
            
    else:
        # Export single model
        if not args.model_path:
            parser.error("--model_path is required when not using --export_all")
        
        if not args.model_type:
            parser.error("--model_type is required when not using --export_all")
        
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return
        
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint_info = infer_checkpoint_info(checkpoint, args.model_type)
        model = build_model(args.model_type, checkpoint_info)
        model.load_state_dict(get_state_dict(checkpoint))
        
        # Export to ONNX
        onnx_path = output_dir / f"{model_path.stem}.onnx"
        export_pytorch_to_onnx(
            model,
            onnx_path,
            model_path.stem,
            checkpoint_info["input_channels"],
            checkpoint_info["input_length"],
        )
        
        # Create metadata
        create_model_metadata(model_path, output_dir, model, args.model_type, checkpoint_info)
        
        # Generate conversion script
        generate_conversion_script(onnx_path, output_dir)
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install conversion tools (if not already installed):")
    print("   pip install onnx-tf tensorflowjs")
    print("\n2. Run the generated conversion script:")
    print(f"   bash {output_dir}/convert_to_tfjs.sh")
    print("\n3. Copy the converted model to your web demo:")
    print(f"   cp -r {output_dir}/<model_name> /path/to/browser_demo/models/")


if __name__ == "__main__":
    main()
