import React, { useState } from 'react';
import { GlossaryModal, GlossaryTerm } from './components/GlossaryModal';
import { WaveformSection } from './components/WaveformSection';
import { CNNSection } from './components/CNNSection';
import { SingleEventView } from './components/SingleEventView';
import './App.css';

function ExplainerTab() {
  const [glossaryTerm, setGlossaryTerm] = useState(null);

  return (
    <>
      <article className="copy-block">
        <h2>What is a Convolutional Neural Network?</h2>
        <p>
          In machine learning, a <GlossaryTerm termKey="classifier" onOpen={setGlossaryTerm}>classifier</GlossaryTerm> assigns a class label to a data point. For example, an image classifier produces a label (e.g. bird, plane) for what's in an image. A <GlossaryTerm termKey="cnn" onOpen={setGlossaryTerm}>convolutional neural network</GlossaryTerm>, or <GlossaryTerm termKey="cnn" onOpen={setGlossaryTerm}>CNN</GlossaryTerm>, is a type of classifier that excels at finding patterns in data.
        </p>
        <figure className="copy-block-figure">
          <img src="/images/cnn-patterns-diagram.png" alt="Diagram: CNN takes an input (e.g. image), processes it through layers to find patterns, and outputs a classification. Below: pipeline from Camera to Preprocessing to CNN to Postprocessing." />
        </figure>
        <p>
          Here we use a CNN not for images but for <GlossaryTerm termKey="seismic waveforms" onOpen={setGlossaryTerm}>seismic waveforms</GlossaryTerm>: the network takes a 1D signal (ground motion over time) and classifies it as <GlossaryTerm termKey="classes" onOpen={setGlossaryTerm}>Noise</GlossaryTerm> or <GlossaryTerm termKey="classes" onOpen={setGlossaryTerm}>Earthquake</GlossaryTerm>. The same ideas—convolutions, layers, learned weights—apply; we just work with 1D time series instead of 2D images.
        </p>
        <p>
          A CNN is a <GlossaryTerm termKey="neural network" onOpen={setGlossaryTerm}>neural network</GlossaryTerm>: an algorithm that recognizes patterns using many small units (neurons) organized in <GlossaryTerm termKey="layers" onOpen={setGlossaryTerm}>layers</GlossaryTerm>, each with learnable <GlossaryTerm termKey="weights and biases" onOpen={setGlossaryTerm}>weights and biases</GlossaryTerm>. Let's break it down.
        </p>
        <ul>
          <li><GlossaryTerm termKey="tensor" onOpen={setGlossaryTerm}>Tensor</GlossaryTerm> — Think of it as a multi-dimensional array of numbers. In our earthquake CNN, the input is a 1D tensor (one channel × many time samples).</li>
          <li><GlossaryTerm termKey="neuron" onOpen={setGlossaryTerm}>Neuron</GlossaryTerm> — A function that takes several inputs and produces one output. Layers are made of many neurons.</li>
          <li><GlossaryTerm termKey="layer" onOpen={setGlossaryTerm}>Layer</GlossaryTerm> — A collection of neurons that all do the same kind of operation (e.g. convolution, pooling).</li>
          <li><GlossaryTerm termKey="kernel" onOpen={setGlossaryTerm}>Kernel (weights) and bias</GlossaryTerm> — Learned during training. They let the network adapt to the data (e.g. detect P-wave vs noise).</li>
          <li><GlossaryTerm termKey="class scores" onOpen={setGlossaryTerm}>Class scores</GlossaryTerm> — The CNN outputs a score for each class (Noise and Earthquake). The highest score is the prediction.</li>
        </ul>
        <p>
          What makes a CNN special is the <GlossaryTerm termKey="convolutional layer" onOpen={setGlossaryTerm}>convolutional layer</GlossaryTerm>. It slides small filters (kernels) over the input to detect local patterns—like the onset of an earthquake wave or the wiggle of seismic noise. That makes CNNs great for signals and images.
        </p>
        <p>
          In this Explainer you can see how a small CNN classifies seismic waveforms. The architecture (Compact or Standard) uses the same building blocks as bigger networks: convolutions, ReLU, pooling, and fully connected layers—just in 1D and on a smaller scale so it's easier to follow.
        </p>
      </article>

      <GlossaryModal termKey={glossaryTerm} onClose={() => setGlossaryTerm(null)} />

      <section className="section">
        <h2 id="waveform-heading">What do the signals look like?</h2>
        <p className="section-desc">Real earthquake waveforms from the Alaska Seismic Network. Choose a magnitude and hover over the curve for details.</p>
        <WaveformSection />
      </section>

      <section className="section">
        <h2 id="cnn-heading">How the CNN classifies the signal</h2>
        <p className="section-desc">Choose a sample and model, then run the network. Use <strong>Step through</strong> to move layer-by-layer with a visual of the signal at each stage. Click a layer in the diagram for a description.</p>
        <CNNSection />
      </section>

      <section className="section results-section">
        <h2 id="results-heading">Results on many waveforms</h2>
        <p className="section-desc">
          The interactive example above uses a <strong>small CNN</strong> (Compact model, ~9,300 parameters) that runs in your browser so you can try it yourself. The <strong>full pipeline</strong> uses the larger <strong>Standard CNN</strong> (~94,000 parameters, 4 conv blocks, 128 feature channels) and is trained on <strong>9,000+ labeled waveforms</strong> from the Alaska Seismic Network. The grid below shows nine example predictions from the Compact model on held-out test data—green means the prediction matched the true label, red means it was wrong (e.g. an earthquake misclassified as noise).
        </p>
        <div className="results-grid">
          <figure className="results-figure">
            <img src="/images/test-predictions-grid.png" alt="Test predictions: 9 waveforms with True label, Predicted label, and confidence. Green = correct, red = incorrect." />
            <figcaption>Test predictions — Compact model (green = correct, red = incorrect). Each plot shows one waveform with true label, predicted label, and confidence.</figcaption>
          </figure>
          <figure className="results-figure">
            <img src="/images/training-validation-curves.png" alt="Training and validation loss and accuracy for Compact vs Standard models over epochs." />
            <figcaption>Training and validation loss (left) and accuracy (right). Compact reaches very low loss and 100% validation accuracy; Standard is larger and behaves differently.</figcaption>
          </figure>
        </div>
      </section>



      <article className="copy-block">
        <h2>Understanding hyperparameters (INPROGRESS)</h2>
        <p>Hyperparameters are settings chosen before training (not learned from data). They shape the network and how it learns.</p>
        <p><strong>Kernel size</strong> — How many time samples each filter sees. In our Compact model: Conv1 uses size 7, Conv2 uses 5, Conv3 uses 3. Smaller kernels catch fine details; larger ones see broader patterns.</p>
        <p><strong>Stride</strong> — How many samples the kernel moves each step. Stride 1 keeps more detail; stride 2 (used in Conv1) halves the time length and speeds up the network.</p>
        <p><strong>Padding</strong> — Extra samples added at the edges so the output length is easier to control. We use &quot;same&quot;-style padding so the signal length is preserved (or reduced only by stride).</p>
        <p><strong>Number of filters</strong> — Each conv layer has a set number of output channels (Compact: 16 → 32 → 64). More filters mean more learned patterns but more parameters and compute.</p>
        <p><strong>Pool size</strong> — Max-pooling uses windows of size 2: we take the maximum over every 2 consecutive samples, so the signal length is halved after each pool.</p>
      </article>
    </>
  );
}

function App() {
  const [tab, setTab] = useState('explainer');

  return (
    <main className="app">
      <header className="app-header">
        <h1>How a CNN detects Earthquakes</h1>
        <p>Learn how convolutional neural networks are used to tell earthquakes from noise.</p>
        <nav className="tabs" aria-label="Main sections">
          <button type="button" className={`tab-btn ${tab === 'explainer' ? 'active' : ''}`} onClick={() => setTab('explainer')}>Explainer</button>
          <button type="button" className={`tab-btn ${tab === 'single' ? 'active' : ''}`} onClick={() => setTab('single')}>Single event</button>
        </nav>
      </header>

      {tab === 'single' ? (
        <section className="section">
          <SingleEventView />
        </section>
      ) : (
        <ExplainerTab />
      )}

      <footer className="footer">
        Made by Derek Yao
      </footer>
    </main>
  );
}

export default App;
