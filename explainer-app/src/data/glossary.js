export const glossary = {
  classifier: {
    title: 'Classifier',
    body: 'A <strong>classifier</strong> is a program that looks at a piece of data and picks a label for it. Our CNN is a classifier that looks at a waveform and says Noise or Earthquake.',
  },
  cnn: {
    title: 'CNN',
    body: 'A <strong>convolutional neural network</strong> (CNN) is a type of classifier that is really good at spotting patterns. It uses special layers called convolutions that slide small filters over the data to find shapes and changes—like the sudden start of an earthquake wave.',
  },
  'seismic waveforms': {
    title: 'Seismic waveforms',
    body: 'A <strong>seismic waveform</strong> is a record of how the ground moved over time. The CNN takes this 1D signal and classifies it.',
  },
  classes: {
    title: 'Noise and Earthquake',
    body: 'These are the two <strong>classes</strong> our CNN can choose from. The CNN gives a score to each and picks the highest.',
  },
  'neural network': {
    title: 'Neural network',
    body: 'A <strong>neural network</strong> is an algorithm made of many small units (neurons) connected in layers. Together they learn to recognize patterns in data.',
  },
  layers: {
    title: 'Layers',
    body: 'A <strong>layer</strong> is one step in the network. Each layer is a group of neurons that do the same kind of job—for example, convolution or pooling.',
  },
  'weights and biases': {
    title: 'Weights and biases',
    body: '<strong>Weights</strong> and <strong>biases</strong> are numbers the network learns during training. They let the network adapt to real data.',
  },
  tensor: {
    title: 'Tensor',
    body: 'A <strong>tensor</strong> is a box of numbers—like a list or table. In our earthquake CNN, the input is a 1D tensor: one channel × many time samples.',
  },
  neuron: {
    title: 'Neuron',
    body: 'A <strong>neuron</strong> takes several numbers as input, multiplies them by learned weights, adds a bias, and outputs one number.',
  },
  layer: {
    title: 'Layer',
    body: 'A <strong>layer</strong> is a set of neurons that all do the same type of operation.',
  },
  kernel: {
    title: 'Kernel (weights) and bias',
    body: 'The <strong>kernel</strong> is the small set of weights slid over the input in a convolution. The <strong>bias</strong> is an extra number added after. Both are learned during training.',
  },
  'class scores': {
    title: 'Class scores',
    body: '<strong>Class scores</strong> are the numbers the CNN outputs for each label. The highest score wins—that\'s the prediction.',
  },
  'convolutional layer': {
    title: 'Convolutional layer',
    body: 'A <strong>convolutional layer</strong> slides small filters (kernels) over the input to detect local patterns—like the onset of an earthquake wave.',
  },
};
