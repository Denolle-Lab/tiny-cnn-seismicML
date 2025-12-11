# Browser Deployment Guide

Complete guide for deploying the seismic CNN classifier in web browsers for K-12 education and block-coding integration.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Export Pipeline](#export-pipeline)
4. [Browser Demo](#browser-demo)
5. [Block-Coding Integration](#block-coding-integration)
6. [Deployment Options](#deployment-options)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Overview

This deployment package enables the seismic CNN classifier to run entirely in web browsers using TensorFlow.js. No server-side computation is required - all predictions happen on the client device.

### Why Browser Deployment?

- ✅ **No Installation**: Works immediately in any modern browser
- ✅ **Universal Access**: Compatible with Chromebooks, tablets, and smartphones
- ✅ **Privacy**: All processing happens locally (no data sent to servers)
- ✅ **Education-Friendly**: Perfect for K-12 classrooms with limited resources
- ✅ **Block-Coding Ready**: Integrates with Scratch, Blockly, and similar platforms

### Technology Stack

```
PyTorch (Training) → ONNX (Intermediate) → TensorFlow.js (Browser)
```

**Key Components**:
- **Python Export Script**: `scripts/export_to_browser.py`
- **JavaScript Wrapper**: `browser_demo/seismic_classifier.js`
- **Web Interface**: `browser_demo/index.html`
- **External Dependencies**: TensorFlow.js 4.11.0, Chart.js 4.4.0

## Architecture

### Model Conversion Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Pipeline                          │
│  PyTorch Model (.pth) → Trained on seismic data            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Export to ONNX                            │
│  torch.onnx.export() → seismic_cnn_*.onnx                  │
│  • Input: [batch, 1, 500]                                   │
│  • Output: [batch, 3] (Noise, Traffic, Earthquake)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Convert to TensorFlow                          │
│  onnx-tf convert → SavedModel format                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Convert to TensorFlow.js                           │
│  tensorflowjs_converter → model.json + shards               │
│  • Graph model format                                        │
│  • Optimized for browser execution                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Browser Deployment                            │
│  Load with tf.loadGraphModel()                              │
│  • Runs on WebGL/WebAssembly                                │
│  • Client-side inference                                    │
└─────────────────────────────────────────────────────────────┘
```

### Browser Architecture

```javascript
// High-level workflow
User Input (Data) 
    ↓
Preprocessing (JavaScript)
    ↓
TensorFlow.js Model
    ↓
Predictions (Probabilities)
    ↓
Visualization (Chart.js)
```

## Export Pipeline

### Step 1: Export Models to ONNX

Use the provided export script:

```bash
# Export all trained models
python scripts/export_to_browser.py --export_all --output_dir browser_demo/models

# Or export a specific model
python scripts/export_to_browser.py \
    --model_path trained_models/seismic_cnn_compact_20241211.pth \
    --model_type compact \
    --output_dir browser_demo/models
```

**What this does**:
1. Loads PyTorch model from `.pth` file
2. Creates dummy input tensor (1, 1, 500)
3. Exports to ONNX format with opset version 12
4. Generates metadata JSON with model info
5. Creates conversion script for next step

**Output Files**:
- `seismic_cnn_*.onnx` - ONNX model
- `seismic_cnn_*_metadata.json` - Model metadata
- `convert_to_tfjs.sh` - Conversion script

### Step 2: Install Conversion Tools

```bash
pip install onnx-tf tensorflowjs
```

**Package Versions** (tested):
- `onnx-tf>=1.10.0`
- `tensorflowjs>=4.11.0`

### Step 3: Convert to TensorFlow.js

Run the auto-generated conversion script:

```bash
cd browser_demo/models
bash convert_to_tfjs.sh
```

Or manually:

```bash
# Step 3a: ONNX → TensorFlow SavedModel
onnx-tf convert \
    -i seismic_cnn_compact.onnx \
    -o seismic_cnn_compact_tf

# Step 3b: TensorFlow → TensorFlow.js
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    seismic_cnn_compact_tf \
    seismic_cnn_compact
```

**Final Output**:
```
browser_demo/models/
├── seismic_cnn_compact/
│   ├── model.json              # Model architecture and weights manifest
│   └── group1-shard1of1.bin    # Model weights (binary)
└── seismic_cnn_compact_metadata.json
```

### Verification

Test the converted model:

```javascript
// In browser console
const model = await tf.loadGraphModel('models/seismic_cnn_compact/model.json');
const input = tf.zeros([1, 1, 500]);
const output = model.predict(input);
console.log(output.shape); // Should be [1, 3]
```

## Browser Demo

### File Structure

```
browser_demo/
├── index.html                 # Main demo interface
├── seismic_classifier.js     # Classifier wrapper class
├── README.md                  # Quick start guide
└── models/                    # Converted models
```

### Key Features

**1. Model Loading**
```javascript
const classifier = new SeismicClassifier();
await classifier.loadModel('models/seismic_cnn_compact/model.json');
```

**2. Preprocessing Pipeline**
- Detrending (mean removal)
- Bandpass filtering (1-45 Hz approximation)
- Normalization (zero mean, unit variance)

**3. Prediction**
```javascript
const waveform = new Float32Array(500); // 5 seconds at 100 Hz
const result = await classifier.predict(waveform);
// result = {
//     predictedClass: 2,
//     className: "Earthquake",
//     confidence: 0.95,
//     probabilities: [0.02, 0.03, 0.95]
// }
```

**4. Visualization**
- Real-time waveform plotting with Chart.js
- Probability bars for all classes
- Color-coded results

### Customization

**Change Colors**:
```css
/* In index.html <style> section */
.button {
    background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
}
```

**Add New Signal Types**:
```javascript
// In seismic_classifier.js
generateSyntheticSignal(signalType) {
    // Add your custom signal generation
    if (signalType === 'custom') {
        // Custom logic here
    }
}
```

**Custom Data Format**:
```javascript
// In index.html handleFileUpload()
if (file.name.endsWith('.custom')) {
    // Add custom parser
}
```

## Block-Coding Integration

### Blockly Integration

Create custom blocks for seismic classification:

```javascript
// Define custom Blockly block
Blockly.Blocks['seismic_classify'] = {
    init: function() {
        this.appendValueInput("WAVEFORM")
            .setCheck("Array")
            .appendField("classify seismic signal");
        this.setOutput(true, "String");
        this.setColour(230);
        this.setTooltip("Classify a seismic waveform");
    }
};

// JavaScript code generator
Blockly.JavaScript['seismic_classify'] = function(block) {
    var waveform = Blockly.JavaScript.valueToCode(block, 'WAVEFORM', 
        Blockly.JavaScript.ORDER_ATOMIC);
    var code = `(await classifier.predict(${waveform})).className`;
    return [code, Blockly.JavaScript.ORDER_FUNCTION_CALL];
};
```

**Example Blocks**:
1. **Load Model** - Initialize classifier
2. **Get Raspberry Shake Data** - Fetch from sensor
3. **Classify Signal** - Run prediction
4. **Get Confidence** - Extract probability
5. **Visualize Result** - Display on screen

### Scratch Integration

For Scratch 3.0, create a custom extension:

```javascript
class SeismicExtension {
    getInfo() {
        return {
            id: 'seismicClassifier',
            name: 'Seismic Classifier',
            blocks: [
                {
                    opcode: 'loadModel',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'load [MODEL_TYPE] model',
                    arguments: {
                        MODEL_TYPE: {
                            type: Scratch.ArgumentType.STRING,
                            menu: 'modelTypes',
                            defaultValue: 'compact'
                        }
                    }
                },
                {
                    opcode: 'classify',
                    blockType: Scratch.BlockType.REPORTER,
                    text: 'classify signal [DATA]',
                    arguments: {
                        DATA: {
                            type: Scratch.ArgumentType.STRING
                        }
                    }
                }
            ],
            menus: {
                modelTypes: {
                    acceptReporters: true,
                    items: ['compact', 'standard']
                }
            }
        };
    }

    async loadModel(args) {
        const classifier = new SeismicClassifier();
        await classifier.loadModel(`models/seismic_cnn_${args.MODEL_TYPE}/model.json`);
        this.classifier = classifier;
    }

    async classify(args) {
        if (!this.classifier) {
            return 'Model not loaded';
        }
        const data = JSON.parse(args.DATA);
        const result = await this.classifier.predict(data);
        return result.className;
    }
}

Scratch.extensions.register(new SeismicExtension());
```

### Raspberry Shake Integration

Fetch data from Raspberry Shake sensors in JavaScript:

```javascript
async function fetchRaspberryShakeData(station, startTime, endTime) {
    const url = `https://fdsnws.raspberryshakedata.com/fdsnws/dataselect/1/query`;
    const params = new URLSearchParams({
        net: 'AM',
        sta: station,
        loc: '00',
        cha: 'EHZ',
        start: startTime.toISOString(),
        end: endTime.toISOString(),
        format: 'json'
    });

    const response = await fetch(`${url}?${params}`);
    const data = await response.json();
    
    // Convert to Float32Array for classifier
    return new Float32Array(data.samples);
}

// Usage in Blockly/Scratch
const waveform = await fetchRaspberryShakeData('R1C3A', 
    new Date('2024-01-01T00:00:00'), 
    new Date('2024-01-01T00:00:05'));
const result = await classifier.predict(waveform);
```

### Educational Activities

**Activity 1: Earthquake Detective**
- Students load real Raspberry Shake data
- Classify signals from their local area
- Create a log of detected earthquakes vs traffic

**Activity 2: Signal Builder**
- Use sliders to create custom waveforms
- Predict what the AI will classify them as
- Learn about signal characteristics

**Activity 3: Model Comparison**
- Load both compact and standard models
- Compare predictions and confidence scores
- Discuss speed vs accuracy tradeoffs

## Deployment Options

### Option 1: GitHub Pages (Free)

1. Push `browser_demo/` to GitHub repository
2. Enable GitHub Pages in repository settings
3. Set source to `main` branch, `/browser_demo` folder
4. Access at: `https://username.github.io/repo-name/`

**Pros**: Free, easy, automatic HTTPS  
**Cons**: Public repositories only (for free tier)

### Option 2: Netlify (Free Tier Available)

1. Create `netlify.toml`:
```toml
[build]
  publish = "browser_demo"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

2. Deploy:
```bash
cd browser_demo
netlify deploy --prod
```

**Pros**: Custom domains, continuous deployment, analytics  
**Cons**: Bandwidth limits on free tier

### Option 3: Vercel (Free Tier Available)

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
cd browser_demo
vercel --prod
```

**Pros**: Fast CDN, serverless functions available  
**Cons**: Execution time limits

### Option 4: Self-Hosted

For schools with existing web servers:

```bash
# Copy files to web server
scp -r browser_demo/ user@server:/var/www/html/seismic-classifier/

# Ensure proper permissions
chmod -R 755 /var/www/html/seismic-classifier/
```

**Nginx configuration**:
```nginx
server {
    listen 80;
    server_name seismic.yourschool.edu;
    root /var/www/html/seismic-classifier;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Cache model files
    location ~* \.(json|bin)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Performance Optimization

### Model Selection

| Model | Parameters | Size | Speed | Accuracy | Best For |
|-------|-----------|------|-------|----------|----------|
| Compact | 20K | 120KB | Fast | Good | Chromebooks, tablets, demos |
| Standard | 100K | 600KB | Medium | Better | Desktops, research |

### Browser Optimizations

**1. Preload Model**
```html
<link rel="preload" href="models/seismic_cnn_compact/model.json" as="fetch">
```

**2. Use Web Workers**
```javascript
// worker.js
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
self.addEventListener('message', async (e) => {
    const {waveform} = e.data;
    const result = await classifier.predict(waveform);
    self.postMessage(result);
});
```

**3. Enable WebGL Acceleration**
```javascript
// Ensure WebGL backend is used
await tf.setBackend('webgl');
console.log('Backend:', tf.getBackend()); // Should be 'webgl'
```

**4. Batch Processing**
```javascript
// Process multiple waveforms at once
const batch = tf.stack(waveforms.map(w => 
    classifier.preprocessWaveform(w)
));
const predictions = model.predict(batch);
```

### Caching Strategy

```javascript
// Service Worker for offline support
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('seismic-v1').then((cache) => {
            return cache.addAll([
                '/',
                '/index.html',
                '/seismic_classifier.js',
                '/models/seismic_cnn_compact/model.json',
                '/models/seismic_cnn_compact/group1-shard1of1.bin'
            ]);
        })
    );
});
```

## Troubleshooting

### Common Issues

**1. Model Loading Fails**

*Symptom*: `Failed to fetch` error in console

*Solutions*:
- Verify files exist in `browser_demo/models/`
- Check that you're using a web server (not `file://`)
- Inspect Network tab for 404 errors
- Ensure CORS headers if hosting remotely

**2. Slow Predictions**

*Symptom*: Takes >5 seconds per prediction

*Solutions*:
- Use compact model instead of standard
- Check browser backend: `console.log(tf.getBackend())`
- Should be 'webgl', not 'cpu'
- Close other tabs/applications
- Try different browser (Chrome recommended)

**3. Incorrect Predictions**

*Symptom*: Always predicts same class

*Solutions*:
- Verify preprocessing matches training
- Check input shape: `console.log(input.shape)`
- Ensure data is Float32Array, not strings
- Validate sampling rate (100 Hz)

**4. Memory Leaks**

*Symptom*: Browser slows down over time

*Solutions*:
- Dispose tensors after use: `tensor.dispose()`
- Use `tf.tidy()` for operations
- Monitor memory: `console.log(tf.memory())`

**5. CORS Errors**

*Symptom*: `Cross-Origin Request Blocked`

*Solutions*:
- Use local server, not file:// protocol
- Add CORS headers if hosting remotely:
```javascript
// Express.js example
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    next();
});
```

### Debug Mode

Enable verbose logging:

```javascript
// In index.html, before loading classifier
tf.ENV.set('DEBUG', true);

// Check TensorFlow.js info
console.log('TF.js version:', tf.version);
console.log('Backend:', tf.getBackend());
console.log('Flags:', tf.ENV.features);
```

### Testing Checklist

- [ ] Models converted successfully
- [ ] Web server running (not file://)
- [ ] Model loads without errors
- [ ] Synthetic signals work
- [ ] File upload works
- [ ] Predictions are reasonable
- [ ] Visualizations display correctly
- [ ] Works on target devices (Chromebooks, etc.)
- [ ] Performance is acceptable
- [ ] No console errors

## Next Steps

1. **Test Deployment**: Verify all features work in target environment
2. **Gather Feedback**: Test with K-12 students/teachers
3. **Optimize**: Profile and improve performance as needed
4. **Scale**: Deploy to production hosting
5. **Monitor**: Track usage and errors in production

## Additional Resources

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Blockly Developer Docs](https://developers.google.com/blockly)
- [Scratch Extensions](https://scratch.mit.edu/developers)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Raspberry Shake API](https://manual.raspberryshake.org/)

## Support

For issues specific to:
- **Model Export**: Check `scripts/export_to_browser.py`
- **Browser Demo**: Review `browser_demo/README.md`
- **Block-Coding**: See integration examples above
- **Deployment**: Consult hosting provider documentation

## License

Same as main project - see LICENSE file.
