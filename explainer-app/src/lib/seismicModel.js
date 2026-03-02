/**
 * Compact seismic CNN in TensorFlow.js.
 * Mirrors PyTorch CompactSeismicCNN: conv1 → bn1 → pool1 → conv2 → bn2 → pool2 → conv3 → bn3 → globalPool → fc.
 * Input: [batch, 6000, 1] (channelsLast). Weights loaded from compact_weights.json.
 */

import * as tf from '@tensorflow/tfjs';

const INPUT_LENGTH = 6000;
const SR = 100;

/**
 * Build the compact model (no weights). Call loadWeightsFromJson after.
 * @param {number} numClasses 2 or 3
 * @returns {tf.LayersModel}
 */
export function buildCompactModel(numClasses = 2) {
  const model = tf.sequential({ name: 'CompactSeismicCNN' });

  // Conv1: 1 → 16, k7, s2, padding 3 → (6000+6-7)/2+1 = 3000
  model.add(tf.layers.conv1d({
    filters: 16,
    kernelSize: 7,
    strides: 2,
    padding: 'same',
    inputShape: [INPUT_LENGTH, 1],
    name: 'conv1',
  }));
  model.add(tf.layers.batchNormalization({ name: 'bn1' }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2, name: 'pool1' }));

  // Conv2: 16 → 32, k5, s1
  model.add(tf.layers.conv1d({
    filters: 32,
    kernelSize: 5,
    strides: 1,
    padding: 'same',
    name: 'conv2',
  }));
  model.add(tf.layers.batchNormalization({ name: 'bn2' }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2, strides: 2, name: 'pool2' }));

  // Conv3: 32 → 64, k3, s1
  model.add(tf.layers.conv1d({
    filters: 64,
    kernelSize: 3,
    strides: 1,
    padding: 'same',
    name: 'conv3',
  }));
  model.add(tf.layers.batchNormalization({ name: 'bn3' }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.globalAveragePooling1d({ name: 'global_pool' }));

  model.add(tf.layers.dense({ units: numClasses, name: 'fc' }));

  return model;
}

/**
 * Load weights from our custom JSON (from export_compact_weights_for_tfjs.py).
 * Sets weights per layer via setWeights().
 * @param {tf.LayersModel} model
 * @param {Record<string, number[][]|number[]>} weightsJson
 */
export function loadWeightsFromJson(model, weightsJson) {
  const setConv = (layerName, kernelKey, biasKey) => {
    model.getLayer(layerName).setWeights([
      tf.tensor(weightsJson[kernelKey]),
      tf.tensor1d(weightsJson[biasKey]),
    ]);
  };

  const setBn = (layerName, prefix) => {
    model.getLayer(layerName).setWeights([
      tf.tensor1d(weightsJson[`${prefix}/gamma`]),
      tf.tensor1d(weightsJson[`${prefix}/beta`]),
      tf.tensor1d(weightsJson[`${prefix}/moving_mean`]),
      tf.tensor1d(weightsJson[`${prefix}/moving_variance`]),
    ]);
  };

  setConv('conv1', 'conv1/kernel', 'conv1/bias');
  setBn('bn1', 'bn1');
  setConv('conv2', 'conv2/kernel', 'conv2/bias');
  setBn('bn2', 'bn2');
  setConv('conv3', 'conv3/kernel', 'conv3/bias');
  setBn('bn3', 'bn3');

  model.getLayer('fc').setWeights([
    tf.tensor(weightsJson['fc/kernel']),
    tf.tensor1d(weightsJson['fc/bias']),
  ]);
}

/**
 * Preprocess waveform: detrend (subtract mean), normalize (zero mean, unit var), then shape [1, 6000, 1].
 * @param {number[]|Float32Array} waveform length 6000
 * @returns {tf.Tensor}
 */
export function preprocessWaveform(waveform) {
  return tf.tidy(() => {
    let t = tf.tensor1d(waveform instanceof Float32Array ? Array.from(waveform) : waveform);
    if (t.shape[0] !== INPUT_LENGTH) {
      if (t.shape[0] < INPUT_LENGTH) {
        const pad = tf.zeros([INPUT_LENGTH - t.shape[0]]);
        t = tf.concat([t, pad]);
      } else {
        t = t.slice([0], [INPUT_LENGTH]);
      }
    }
    const mean = t.mean();
    t = t.sub(mean);
    const variance = tf.moments(t).variance;
    t = t.div(tf.sqrt(variance).add(1e-8));
    return t.reshape([1, INPUT_LENGTH, 1]);
  });
}

/**
 * Run model and return probabilities + predicted class index.
 * @param {tf.LayersModel} model
 * @param {number[]|Float32Array} waveform
 * @param {string[]} classNames
 * @returns {{ predictedClass: number, className: string, confidence: number, probabilities: number[], classNames: string[] }}
 */
export function predict(model, waveform, classNames) {
  return tf.tidy(() => {
    const input = preprocessWaveform(waveform);
    const logits = model.predict(input);
    const probs = tf.softmax(logits);
    const probArray = Array.from(probs.dataSync());
    const predIdx = probArray.indexOf(Math.max(...probArray));
    return {
      predictedClass: predIdx,
      className: classNames[predIdx] ?? `Class ${predIdx}`,
      confidence: probArray[predIdx],
      probabilities: probArray,
      classNames,
    };
  });
}

/**
 * Step definitions for the Compact model: stepId, layerIndex (-1 = input), and short label.
 * Layer indices: 0 conv1, 1 bn1, 2 relu, 3 pool1, 4 conv2, 5 bn2, 6 relu, 7 pool2, 8 conv3, 9 bn3, 10 relu, 11 global_pool, 12 fc.
 */
export const COMPACT_STEPS = [
  { stepId: 'input', layerIndex: -1, label: 'Input (raw waveform)' },
  { stepId: 'conv1', layerIndex: 2, label: 'After Conv1 + ReLU' },
  { stepId: 'pool1', layerIndex: 3, label: 'After Pool 1' },
  { stepId: 'conv2', layerIndex: 6, label: 'After Conv2 + ReLU' },
  { stepId: 'pool2', layerIndex: 7, label: 'After Pool 2' },
  { stepId: 'conv3', layerIndex: 10, label: 'After Conv3 + ReLU' },
  { stepId: 'global_pool', layerIndex: 11, label: 'After global average pool' },
  { stepId: 'fc', layerIndex: 12, label: 'Class scores (FC output)' },
];

/**
 * Run waveform through the model and return activations at each step for visualization.
 * Applies layers manually so we don't rely on model.input or slice models.
 * @param {tf.LayersModel} model
 * @param {number[]|Float32Array} waveform
 * @returns {{ stepId: string, label: string, data: number[], is1D: boolean }[]}
 */
export function getActivationsAtSteps(model, waveform) {
  return tf.tidy(() => {
    let x = preprocessWaveform(waveform);
    const steps = [];
    const stepLayerIndices = COMPACT_STEPS.map((s) => s.layerIndex);

    if (COMPACT_STEPS[0].layerIndex === -1) {
      steps.push({
        stepId: 'input',
        label: COMPACT_STEPS[0].label,
        data: Array.from(x.dataSync()),
        is1D: true,
      });
    }

    let lastSavedIndex = -1;
    for (let layerIdx = 0; layerIdx < model.layers.length; layerIdx++) {
      x = model.layers[layerIdx].apply(x);
      const stepIdx = stepLayerIndices.indexOf(layerIdx);
      if (stepIdx === -1) continue;
      const step = COMPACT_STEPS[stepIdx];
      const shape = x.shape;
      const arr = Array.from(x.dataSync());

      if (shape.length === 3) {
        const [, time, ch] = shape;
        const oneChannel = [];
        for (let i = 0; i < time; i++) {
          let sum = 0;
          for (let c = 0; c < ch; c++) sum += arr[i * ch + c];
          oneChannel.push(sum / ch);
        }
        steps.push({ stepId: step.stepId, label: step.label, data: oneChannel, is1D: true });
      } else {
        steps.push({ stepId: step.stepId, label: step.label, data: arr, is1D: false });
      }
      lastSavedIndex = layerIdx;
    }

    return steps;
  });
}

export { INPUT_LENGTH, SR };
