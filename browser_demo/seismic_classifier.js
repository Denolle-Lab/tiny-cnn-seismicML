/**
 * Seismic Signal Classifier - JavaScript Wrapper for TensorFlow.js
 * 
 * This module provides a browser-compatible interface for the seismic CNN model.
 * It handles model loading, preprocessing, and prediction using TensorFlow.js.
 * 
 * Usage:
 *   const classifier = new SeismicClassifier();
 *   await classifier.loadModel('models/seismic_cnn_compact/model.json');
 *   const predictions = await classifier.predict(waveformData);
 * 
 * @author Seismic ML Project
 * @version 1.0.0
 */

class SeismicClassifier {
    constructor() {
        this.model = null;
        this.metadata = null;
        this.classNames = ['Noise', 'Traffic', 'Earthquake'];
        this.samplingRate = 100; // Hz
        this.windowDuration = 5.0; // seconds
        this.expectedSamples = 500; // 5s * 100Hz
    }

    /**
     * Load the TensorFlow.js model and metadata
     * @param {string} modelPath - Path to model.json file
     * @param {string} metadataPath - Path to metadata JSON (optional)
     */
    async loadModel(modelPath, metadataPath = null) {
        try {
            console.log('Loading model from:', modelPath);
            this.model = await tf.loadGraphModel(modelPath);
            console.log('✓ Model loaded successfully');

            // Load metadata if provided
            if (metadataPath) {
                const response = await fetch(metadataPath);
                this.metadata = await response.json();
                console.log('✓ Metadata loaded:', this.metadata);
                
                // Update parameters from metadata
                this.classNames = this.metadata.class_names || this.classNames;
                this.samplingRate = this.metadata.sampling_rate || this.samplingRate;
                this.windowDuration = this.metadata.window_duration || this.windowDuration;
            }

            // Warm up the model with a dummy prediction
            await this.warmup();
            
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    /**
     * Warm up the model by running a dummy prediction
     */
    async warmup() {
        const dummyInput = tf.zeros([1, 1, this.expectedSamples]);
        const prediction = await this.model.predict(dummyInput);
        prediction.dispose();
        dummyInput.dispose();
        console.log('✓ Model warmup complete');
    }

    /**
     * Detrend a signal by removing the mean
     * @param {tf.Tensor} signal - Input signal tensor
     * @returns {tf.Tensor} Detrended signal
     */
    detrend(signal) {
        return tf.tidy(() => {
            const mean = signal.mean();
            return signal.sub(mean);
        });
    }

    /**
     * Apply bandpass filter approximation using FFT
     * Note: This is a simplified version. For production, consider a proper
     * Butterworth filter implementation in JavaScript.
     * 
     * @param {tf.Tensor} signal - Input signal tensor
     * @param {number} freqmin - Minimum frequency (Hz)
     * @param {number} freqmax - Maximum frequency (Hz)
     * @returns {tf.Tensor} Filtered signal
     */
    bandpassFilter(signal, freqmin = 1.0, freqmax = 45.0) {
        // For browser deployment, we use a simplified approach
        // A full Butterworth filter would require additional libraries
        return tf.tidy(() => {
            // Apply high-pass by subtracting rolling mean (simple approximation)
            const windowSize = Math.floor(this.samplingRate / freqmin);
            const smoothed = this.movingAverage(signal, windowSize);
            let filtered = signal.sub(smoothed);
            
            // Normalize
            const std = tf.moments(filtered).variance.sqrt();
            filtered = filtered.div(std.add(1e-8));
            
            return filtered;
        });
    }

    /**
     * Calculate moving average for simple filtering
     * @param {tf.Tensor} signal - Input signal
     * @param {number} windowSize - Window size for averaging
     * @returns {tf.Tensor} Smoothed signal
     */
    movingAverage(signal, windowSize) {
        return tf.tidy(() => {
            const kernel = tf.ones([windowSize]).div(windowSize);
            // Simple convolution-based moving average
            const reshaped = signal.reshape([1, -1, 1]);
            const kernelReshaped = kernel.reshape([windowSize, 1, 1]);
            const smoothed = tf.conv1d(reshaped, kernelReshaped, 1, 'same');
            return smoothed.reshape(signal.shape);
        });
    }

    /**
     * Normalize signal to zero mean and unit variance
     * @param {tf.Tensor} signal - Input signal tensor
     * @returns {tf.Tensor} Normalized signal
     */
    normalize(signal) {
        return tf.tidy(() => {
            const moments = tf.moments(signal);
            const normalized = signal.sub(moments.mean).div(moments.variance.sqrt().add(1e-8));
            return normalized;
        });
    }

    /**
     * Preprocess waveform data to match training pipeline
     * @param {Array|Float32Array|tf.Tensor} waveform - Raw waveform data
     * @returns {tf.Tensor} Preprocessed tensor ready for model input
     */
    preprocessWaveform(waveform) {
        return tf.tidy(() => {
            // Convert to tensor if needed
            let signal = waveform instanceof tf.Tensor ? 
                waveform : tf.tensor1d(waveform);

            // Validate length
            if (signal.shape[0] !== this.expectedSamples) {
                console.warn(`Expected ${this.expectedSamples} samples, got ${signal.shape[0]}`);
                // Pad or truncate as needed
                if (signal.shape[0] < this.expectedSamples) {
                    const padding = tf.zeros([this.expectedSamples - signal.shape[0]]);
                    signal = tf.concat([signal, padding]);
                } else {
                    signal = signal.slice([0], [this.expectedSamples]);
                }
            }

            // Apply preprocessing pipeline
            signal = this.detrend(signal);
            signal = this.bandpassFilter(signal);
            signal = this.normalize(signal);

            // Reshape to model input format: [batch, channels, samples]
            signal = signal.reshape([1, 1, this.expectedSamples]);

            return signal;
        });
    }

    /**
     * Make prediction on a single waveform
     * @param {Array|Float32Array|tf.Tensor} waveform - Raw waveform data
     * @returns {Object} Prediction results with probabilities and predicted class
     */
    async predict(waveform) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        return tf.tidy(() => {
            // Preprocess the waveform
            const processedInput = this.preprocessWaveform(waveform);

            // Run inference
            const predictions = this.model.predict(processedInput);
            
            // Apply softmax to get probabilities
            const probabilities = tf.softmax(predictions);
            
            // Get values as array
            const probArray = Array.from(probabilities.dataSync());
            
            // Find predicted class
            const predictedClass = probArray.indexOf(Math.max(...probArray));
            const confidence = probArray[predictedClass];

            return {
                predictedClass: predictedClass,
                className: this.classNames[predictedClass],
                confidence: confidence,
                probabilities: probArray,
                classNames: this.classNames
            };
        });
    }

    /**
     * Make predictions on multiple waveforms in batch
     * @param {Array} waveforms - Array of waveform data
     * @returns {Array} Array of prediction results
     */
    async predictBatch(waveforms) {
        const results = [];
        for (const waveform of waveforms) {
            const result = await this.predict(waveform);
            results.push(result);
        }
        return results;
    }

    /**
     * Generate synthetic seismic signal for testing
     * @param {string} signalType - Type of signal ('noise', 'traffic', 'earthquake')
     * @returns {Float32Array} Synthetic waveform data
     */
    generateSyntheticSignal(signalType = 'noise') {
        const samples = this.expectedSamples;
        const signal = new Float32Array(samples);
        const dt = 1.0 / this.samplingRate;

        switch (signalType.toLowerCase()) {
            case 'noise':
                // Random noise
                for (let i = 0; i < samples; i++) {
                    signal[i] = (Math.random() - 0.5) * 2.0;
                }
                break;

            case 'traffic':
                // Low frequency oscillation (1-5 Hz)
                const trafficFreq = 2.0 + Math.random() * 3.0;
                for (let i = 0; i < samples; i++) {
                    const t = i * dt;
                    signal[i] = Math.sin(2 * Math.PI * trafficFreq * t) * 
                               (1 + 0.3 * Math.random());
                }
                break;

            case 'earthquake':
                // Higher frequency with amplitude envelope (2-20 Hz)
                const eqFreq = 5.0 + Math.random() * 10.0;
                const centerTime = samples / 2;
                for (let i = 0; i < samples; i++) {
                    const t = i * dt;
                    // Gaussian envelope
                    const envelope = Math.exp(-Math.pow((i - centerTime) / (samples / 4), 2));
                    signal[i] = Math.sin(2 * Math.PI * eqFreq * t) * envelope * 5.0 +
                               (Math.random() - 0.5) * 0.5;
                }
                break;

            default:
                console.warn(`Unknown signal type: ${signalType}, using noise`);
                return this.generateSyntheticSignal('noise');
        }

        return signal;
    }

    /**
     * Get model information
     * @returns {Object} Model metadata and statistics
     */
    getModelInfo() {
        return {
            loaded: this.model !== null,
            metadata: this.metadata,
            classNames: this.classNames,
            samplingRate: this.samplingRate,
            windowDuration: this.windowDuration,
            expectedSamples: this.expectedSamples
        };
    }

    /**
     * Dispose of the model and free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            console.log('✓ Model disposed');
        }
    }
}

// Export for both browser and Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SeismicClassifier;
}
