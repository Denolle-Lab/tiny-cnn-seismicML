# Browser Demo - Seismic Signal Classifier

Interactive web-based demonstration of the seismic CNN classifier using TensorFlow.js.

## Quick Start

### 1. Export and Convert Your Model

First, export your trained PyTorch model to browser-compatible format:

```bash
# Export all trained models to ONNX
python scripts/export_to_browser.py --export_all --output_dir browser_demo/models

# Install conversion tools (if not already installed)
pip install onnx-tf tensorflowjs

# Convert ONNX to TensorFlow.js (use the generated script)
cd browser_demo/models
bash convert_to_tfjs.sh
```

### 2. Set Up the Demo

Your directory structure should look like this:

```
browser_demo/
├── index.html                          # Main demo page
├── seismic_classifier.js              # Classifier wrapper
├── README.md                          # This file
└── models/                            # Model files
    ├── seismic_cnn_compact/          # Compact model (TF.js)
    │   ├── model.json
    │   └── group1-shard1of1.bin
    ├── seismic_cnn_standard/         # Standard model (TF.js)
    │   ├── model.json
    │   └── group1-shard1of1.bin
    ├── seismic_cnn_compact_metadata.json
    └── seismic_cnn_standard_metadata.json
```

### 3. Run the Demo

Start a local web server:

```bash
# Using Python
python -m http.server 8000

# Or using Node.js
npx http-server -p 8000

# Or using PHP
php -S localhost:8000
```

Then open your browser to: http://localhost:8000

## Features

### 🎯 Interactive Demo

- **Model Selection**: Choose between compact (fast) or standard (accurate) models
- **Synthetic Signals**: Test with pre-generated noise, traffic, or earthquake signals
- **Custom Data Upload**: Upload your own seismic data (CSV, TXT, or JSON format)
- **Real-time Visualization**: See waveforms with Chart.js
- **Instant Predictions**: Get classification results with confidence scores

### 📊 Supported Data Formats

**CSV/TXT Format** (one value per line or comma-separated):
```
0.123
-0.456
0.789
...
```

**JSON Format**:
```json
{
  "data": [0.123, -0.456, 0.789, ...],
  "timestamp": "2024-01-01T00:00:00",
  "station": "AM.R1C3A.00.EHZ"
}
```

### 🎓 K-12 Educational Use

Perfect for classroom demonstrations:
- Simple, intuitive interface
- Visual feedback with emojis and colors
- Step-by-step workflow
- Works on Chromebooks and tablets
- No installation required (runs in browser)

## Usage Examples

### Testing with Synthetic Data

1. Click "Load Compact Model (Fast)"
2. Wait for the green indicator: "✓ Compact model loaded successfully"
3. Click one of the test buttons:
   - 🔊 Test Noise Signal
   - 🚗 Test Traffic Signal
   - 🌋 Test Earthquake Signal
4. View the waveform and prediction results

### Using Custom Data

1. Load a model (compact or standard)
2. Click "📁 Upload Your Own Seismic Data"
3. Select a file (CSV, TXT, or JSON)
4. View automatic classification results

### Integration with Raspberry Shake

Export data from your Raspberry Shake sensor:

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import json

# Configure FDSN client for Raspberry Shake data server
client = Client(base_url='https://data.raspberryshake.org')

# Define time window (example: 24 hours on a specific date)
start_time = UTCDateTime("2024-01-01T00:00:00")
end_time = start_time + 24*3600

# Get data from Raspberry Shake
# Network: AM (for Americas), Station: your station code, Channel: EHZ (vertical)
stream = client.get_waveforms(
    network="AM",
    station="R1C3A",  # Replace with your station code
    location="00",
    channel="EHZ",    # Or "EH*" for all channels
    starttime=start_time,
    endtime=end_time
)

print(f"Downloaded {len(stream)} trace(s)")

# Export to JSON for browser demo
data = {
    "data": stream[0].data.tolist(),
    "sampling_rate": stream[0].stats.sampling_rate,
    "station": str(stream[0].stats.station),
    "channel": str(stream[0].stats.channel),
    "starttime": str(stream[0].stats.starttime)
}

with open('seismic_data.json', 'w') as f:
    json.dump(data, f)
```

Then upload `seismic_data.json` to the browser demo!

**Note**: For detailed examples of working with Raspberry Shake data, see the notebooks in `notebooks/01_data_exploration/`.

## Troubleshooting

### Model Not Loading

**Error**: "Failed to fetch model.json"

**Solution**: 
- Make sure you've converted models to TensorFlow.js format
- Check that model files are in `browser_demo/models/`
- Ensure you're running a web server (not opening HTML directly)

### CORS Errors

**Error**: "Cross-Origin Request Blocked"

**Solution**:
- Use a local web server (see "Run the Demo" above)
- Don't open `index.html` directly in browser
- If deploying online, ensure proper CORS headers

### Prediction Errors

**Error**: "Input tensor shape mismatch"

**Solution**:
- Ensure data has exactly 500 samples (5 seconds at 100 Hz)
- Check that data is numeric (not strings)
- Verify sampling rate matches training data (100 Hz)

### Performance Issues

**Problem**: Slow predictions on older devices

**Solution**:
- Use the compact model (20K parameters vs 100K)
- Close other browser tabs
- Try Chrome/Edge (better TensorFlow.js support)

## Advanced Usage

### Programmatic Access

```javascript
// Load classifier
const classifier = new SeismicClassifier();
await classifier.loadModel('models/seismic_cnn_compact/model.json');

// Generate test signal
const waveform = classifier.generateSyntheticSignal('earthquake');

// Make prediction
const result = await classifier.predict(waveform);
console.log('Predicted:', result.className);
console.log('Confidence:', result.confidence);
console.log('All probabilities:', result.probabilities);
```

### Batch Processing

```javascript
const waveforms = [
    classifier.generateSyntheticSignal('noise'),
    classifier.generateSyntheticSignal('traffic'),
    classifier.generateSyntheticSignal('earthquake')
];

const results = await classifier.predictBatch(waveforms);
results.forEach((result, i) => {
    console.log(`Signal ${i}: ${result.className} (${result.confidence.toFixed(2)})`);
});
```

### Custom Preprocessing

The classifier uses the same preprocessing as training:
1. Detrend (remove mean)
2. Bandpass filter (1-45 Hz)
3. Normalize (zero mean, unit variance)

You can access these methods directly:

```javascript
const raw = new Float32Array([...]); // Your raw data
const processed = classifier.preprocessWaveform(raw);
```

## Browser Compatibility

✅ **Supported Browsers**:
- Chrome/Chromium 80+
- Firefox 75+
- Safari 14+
- Edge 80+

⚠️ **Limited Support**:
- Mobile browsers (slower performance)
- Internet Explorer (not supported)

## File Size Considerations

| Model Type | ONNX Size | TF.js Size | Load Time* |
|-----------|-----------|------------|------------|
| Compact   | ~80 KB    | ~120 KB    | 1-2 sec    |
| Standard  | ~400 KB   | ~600 KB    | 2-4 sec    |

*Approximate times on typical broadband connection

## Next Steps

- **Block Coding Integration**: See `docs/BROWSER_DEPLOYMENT.md` for Scratch/Blockly setup
- **Custom Styling**: Modify CSS in `index.html` to match your branding
- **Additional Features**: Add data export, batch processing, or real-time streaming
- **Deploy Online**: Host on GitHub Pages, Netlify, or Vercel

## Support

For questions or issues:
- Check the main project README
- Review `docs/BROWSER_DEPLOYMENT.md` for detailed deployment guide
- Open an issue on GitHub

## License

Same as main project - see LICENSE file.
