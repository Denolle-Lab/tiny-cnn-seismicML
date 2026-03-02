import React, { useState, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useWaveformData } from '../hooks/useWaveformData';
import { useSeismicModel } from '../hooks/useSeismicModel';
import { COMPACT_STEPS } from '../lib/seismicModel';
import './CNNSection.css';

const SR = 100;
const LAYER_DESCRIPTIONS = {
  input: { title: 'Input layer', body: 'The raw waveform: one channel, 6000 samples at 100 Hz (60 seconds). This is what the seismometer recorded.' },
  conv1: { title: 'Convolutional layer 1', body: 'Slides 16 learned filters over the input to detect local patterns—like a sudden jump when the P-wave arrives. ReLU keeps only positive values.' },
  conv2: { title: 'Convolutional layer 2', body: 'A second set of 32 filters runs on the first layer\'s output, catching more complex patterns.' },
  conv3: { title: 'Convolutional layer 3', body: 'Another convolution stage (64 filters). By now the network is responding to patterns that help tell earthquake from noise.' },
  conv4: { title: 'Convolutional layer', body: 'In the Standard model, a fourth conv layer adds more capacity.' },
  pool1: { title: 'Pooling layer 1', body: 'Max-pooling shrinks the signal in time by taking the maximum in each window of 2. Reduces length by half while keeping the strongest activations.' },
  pool2: { title: 'Pooling layer 2', body: 'Same max-pooling again. The signal gets shorter while keeping important activations.' },
  pool3: { title: 'Pooling layer', body: 'Another pooling step.' },
  pool4: { title: 'Pooling layer', body: 'Final pooling before the fully connected layer (Standard model only).' },
  global_pool: { title: 'Global average pooling', body: 'Averages each of the 64 feature channels over time into a single number per channel. The 60-second signal is now 64 numbers—a compact summary for the classifier.' },
  fc: { title: 'Fully connected layer', body: 'Takes the 64 numbers and turns them into class scores (e.g. Noise vs Earthquake). The highest score wins.' },
  output: { title: 'Output', body: 'The class scores. The CNN picks the class with the highest score as the prediction.' },
};

const CLASS_COLORS = { Noise: '#586069', Traffic: '#e36209', Earthquake: '#d73a49' };

const COMPACT_LAYERS = [
  { id: 'input', label: 'Input\n(1×6000)', type: 'input' },
  { id: 'conv1', label: 'Conv1\n(16)', type: 'conv' },
  { id: 'pool1', label: 'Pool', type: 'pool' },
  { id: 'conv2', label: 'Conv2\n(32)', type: 'conv' },
  { id: 'pool2', label: 'Pool', type: 'pool' },
  { id: 'conv3', label: 'Conv3\n(64)', type: 'conv' },
  { id: 'fc', label: 'FC → 2', type: 'fc' },
  { id: 'output', label: 'Output', type: 'output' },
];

const STANDARD_LAYERS = [
  { id: 'input', label: 'Input\n(1×6000)', type: 'input' },
  { id: 'conv1', label: 'Conv1\n(32)', type: 'conv' },
  { id: 'pool1', label: 'Pool', type: 'pool' },
  { id: 'conv2', label: 'Conv2\n(64)', type: 'conv' },
  { id: 'pool2', label: 'Pool', type: 'pool' },
  { id: 'conv3', label: 'Conv3\n(128)', type: 'conv' },
  { id: 'pool3', label: 'Pool', type: 'pool' },
  { id: 'conv4', label: 'Conv4\n(128)', type: 'conv' },
  { id: 'pool4', label: 'Pool', type: 'pool' },
  { id: 'fc', label: 'FC → 2', type: 'fc' },
  { id: 'output', label: 'Output', type: 'output' },
];

function drawActivationChart(container, stepData, classNames = null) {
  if (!container || !stepData?.data?.length) return;
  const raw = stepData.data;
  const data = raw.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0));
  if (!data.length) return;

  d3.select(container).selectAll('*').remove();
  const margin = { top: 12, right: 12, bottom: 32, left: 44 };
  const containerWidth = container.clientWidth || 400;
  const width = Math.max(280, containerWidth - margin.left - margin.right);
  const height = Math.max(80, 160 - margin.top - margin.bottom);

  if (stepData.is1D && data.length > 1) {
    const xScale = d3.scaleLinear().domain([0, data.length - 1]).range([0, width]);
    const yExtent = d3.extent(data);
    const yMin = Number.isFinite(yExtent[0]) ? yExtent[0] : 0;
    const yMaxVal = Number.isFinite(yExtent[1]) ? yExtent[1] : 0;
    const pad = Math.max(1e-6, (yMaxVal - yMin) * 0.1) || 0.01;
    const yScale = d3.scaleLinear().domain([yMin - pad, yMaxVal + pad]).range([height, 0]).nice();
    const line = d3.line().x((d, i) => xScale(i)).y((d) => yScale(d)).curve(d3.curveMonotoneX);
    const svg = d3.select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', 160)
      .attr('viewBox', `0 0 ${width + margin.left + margin.right} 160`);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    g.append('path').attr('d', line(data)).attr('fill', 'none').attr('stroke', '#0969da').attr('stroke-width', 1.2).attr('stroke-linejoin', 'round');
    g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
    g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
    svg.append('text').attr('class', 'axis-label').attr('x', margin.left + width / 2).attr('y', 156).attr('text-anchor', 'middle').text('Time step');
  } else {
    const isClassScores = data.length === 2 && classNames && classNames.length === 2;
    const labels = isClassScores ? classNames : d3.range(data.length).map((i) => `F${i}`);
    const xScale = d3.scaleBand().domain(labels).range([0, width]).padding(0.3);
    const yExtent = d3.extent(data);
    const extMin = Number.isFinite(yExtent[0]) ? yExtent[0] : 0;
    const extMax = Number.isFinite(yExtent[1]) ? yExtent[1] : 0;
    const yMax = Math.max(Math.abs(extMin), Math.abs(extMax), 0.01);
    const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([height, 0]);

    const svg = d3.select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', 160)
      .attr('viewBox', `0 0 ${width + margin.left + margin.right} 160`);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    g.selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', (d, i) => xScale(labels[i]))
      .attr('y', (d) => (d >= 0 ? yScale(d) : yScale(0)))
      .attr('width', Math.max(0, xScale.bandwidth()))
      .attr('height', (d) => Math.abs(yScale(d) - yScale(0)))
      .attr('fill', (d, i) => {
        if (isClassScores) return labels[i] === 'Earthquake' ? '#cf222e' : '#586069';
        return d >= 0 ? '#0969da' : '#cf222e';
      });
    const xAxis = d3.axisBottom(xScale).tickSizeOuter(0);
    if (labels.length > 16) {
      const step = Math.max(1, Math.floor(labels.length / 8));
      xAxis.tickValues(labels.filter((_, i) => i % step === 0));
    }
    g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${height})`).call(xAxis);
    g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
    svg.append('text').attr('class', 'axis-label').attr('x', margin.left + width / 2).attr('y', 156).attr('text-anchor', 'middle').text(isClassScores ? 'Raw score (before softmax)' : 'Feature');
  }
  container.querySelectorAll('.axis path, .axis line').forEach((el) => { el.style.stroke = 'var(--border)'; });
  container.querySelectorAll('.axis text, .axis-label').forEach((el) => { el.style.fill = 'var(--text-muted)'; el.style.fontSize = '10px'; });
}

function drawRunningWaveform(container, data, isNoise) {
  if (!container || !data?.length) return;
  d3.select(container).selectAll('*').remove();
  const len = data.length;
  const timeMax = (len - 1) / SR;
  const margin = { top: 8, right: 12, bottom: 26, left: 44 };
  const width = Math.max(280, container.clientWidth - margin.left - margin.right);
  const height = 128 - margin.top - margin.bottom;
  const xScale = d3.scaleLinear().domain([0, timeMax]).range([0, width]);
  const yExtent = d3.extent(data);
  const pad = Math.max(0.5, (yExtent[1] - yExtent[0]) * 0.15) || 0.5;
  const yScale = d3.scaleLinear().domain([yExtent[0] - pad, yExtent[1] + pad]).range([height, 0]).nice();
  const strokeColor = isNoise ? '#586069' : '#d73a49';
  const line = d3.line().x((d, i) => xScale(i / SR)).y((d) => yScale(d)).curve(d3.curveMonotoneX);
  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('height', 128)
    .attr('viewBox', `0 0 ${width + margin.left + margin.right} 128`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  g.append('path').attr('d', line(data)).attr('fill', 'none').attr('stroke', strokeColor).attr('stroke-width', 1.2).attr('stroke-linejoin', 'round');
  g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
  g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
  svg.append('text').attr('class', 'axis-label').attr('x', margin.left + width / 2).attr('y', 124).attr('text-anchor', 'middle').text('Time (s)');
}

const BOX_W = 102;
const BOX_H = 72;
const BOX_EXPAND = 1.35;
const BOX_GAP = 20;
const STROKE_DEFAULT = '#1f2328';
const STROKE_HIGHLIGHT = '#0969da';

function renderDiagram(container, layers, onLayerClick, highlightedIndex = -1) {
  if (!container) return;
  d3.select(container).selectAll('*').remove();

  const widths = layers.map((_, i) => (i === highlightedIndex ? BOX_W * BOX_EXPAND : BOX_W));
  const heights = layers.map((_, i) => (i === highlightedIndex ? BOX_H * BOX_EXPAND : BOX_H));
  const centers = [];
  let cx = widths[0] / 2;
  for (let i = 0; i < layers.length; i++) {
    centers.push(cx);
    if (i < layers.length - 1) cx += widths[i] / 2 + BOX_GAP + widths[i + 1] / 2;
  }
  const totalW = cx + widths[widths.length - 1] / 2;
  const padding = 24;
  const width = totalW + padding * 2;
  const height = 320;
  const y = height / 2;
  const gOffset = padding;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  svg.append('defs').append('marker')
    .attr('id', 'arrow-cnn')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#1f2328');

  const g = svg.append('g').attr('class', 'layer-group').attr('transform', `translate(${gOffset}, 0)`);
  layers.forEach((layer, i) => {
    const x = centers[i];
    const isHighlighted = i === highlightedIndex;
    const w = widths[i];
    const h = heights[i];

    const box = g.append('g').attr('class', 'layer').attr('data-layer', layer.id).style('cursor', 'pointer');
    const dimmed = highlightedIndex >= 0 && !isHighlighted;
    box.append('rect')
      .attr('class', `layer-box ${layer.type} ${isHighlighted ? 'layer-box-highlighted' : ''} ${dimmed ? 'layer-box-dimmed' : ''}`)
      .attr('x', x - w / 2)
      .attr('y', y - h / 2)
      .attr('width', w)
      .attr('height', h)
      .attr('stroke', isHighlighted ? STROKE_HIGHLIGHT : STROKE_DEFAULT)
      .attr('stroke-width', isHighlighted ? 3 : 2)
      .attr('opacity', dimmed ? 0.55 : 1)
      .on('click', () => onLayerClick(layer.id));
    box.append('text')
      .attr('class', 'layer-label')
      .attr('x', x)
      .attr('y', y)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .text(layer.label)
      .on('click', () => onLayerClick(layer.id));
    if (i < layers.length - 1) {
      g.append('path')
        .attr('class', 'flow-line')
        .attr('d', `M ${x + w / 2} ${y} L ${centers[i + 1] - widths[i + 1] / 2} ${y}`)
        .attr('marker-end', 'url(#arrow-cnn)');
    }
  });
}

export function CNNSection() {
  const { data } = useWaveformData();
  const { ready: modelReady, classNames, predict: runModelPredict, getActivationsAtSteps } = useSeismicModel();
  const [sample, setSample] = useState('small');
  const [model, setModel] = useState('compact');
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [running, setRunning] = useState(false);
  const [stepThroughActive, setStepThroughActive] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [activations, setActivations] = useState(null);
  const runningRef = useRef(null);
  const diagramRef = useRef(null);
  const activationChartRef = useRef(null);

  const runningWaveform = (() => {
    if (!data) return null;
    if (sample === 'noise' && data.noise?.length) return data.noise[0].waveform;
    const eq = data.earthquake?.[sample]?.[0] ?? data.earthquake?.small?.[0] ?? data.earthquake?.medium?.[0] ?? data.earthquake?.large?.[0];
    return eq?.waveform ?? null;
  })();

  const runningMeta = (() => {
    if (!data) return null;
    if (sample === 'noise' && data.noise?.length) return data.noise[0];
    return data.earthquake?.[sample]?.[0] ?? data.earthquake?.small?.[0] ?? data.earthquake?.medium?.[0] ?? data.earthquake?.large?.[0] ?? null;
  })();

  const sampleLabel = sample === 'noise' ? 'Noise' : sample === 'small' ? 'Small magnitude' : sample === 'medium' ? 'Medium magnitude' : 'Large magnitude';
  const layers = model === 'compact' ? COMPACT_LAYERS : STANDARD_LAYERS;

  useEffect(() => {
    if (runningWaveform && runningRef.current) drawRunningWaveform(runningRef.current, runningWaveform, sample === 'noise');
  }, [runningWaveform, sample]);

  const stepIndexToDiagramIndex = model === 'compact'
    ? stepIndex
    : [0, 1, 2, 3, 4, 5, 8, 9][stepIndex]; // Standard: map global_pool→pool4, fc→fc
  const highlightedLayerIndex =
    stepThroughActive && activations?.length
      ? (model === 'compact' ? stepIndex : (stepIndexToDiagramIndex ?? stepIndex))
      : selectedLayer
        ? layers.findIndex((l) => l.id === selectedLayer)
        : -1;

  useEffect(() => {
    if (diagramRef.current) {
      renderDiagram(
        diagramRef.current,
        layers,
        setSelectedLayer,
        highlightedLayerIndex
      );
    }
  }, [model, highlightedLayerIndex]);

  useEffect(() => {
    if (!stepThroughActive || !activations?.length || stepIndex < 0 || stepIndex >= activations.length) return;
    const step = activations[stepIndex];
    if (!step?.data) return;
    const isFcStep = step.stepId === 'fc';
    if (!activationChartRef.current) return;
    try {
      drawActivationChart(activationChartRef.current, step, isFcStep ? classNames : null);
    } catch (err) {
      console.error('drawActivationChart error:', err);
    }
  }, [stepThroughActive, activations, stepIndex, classNames]);

  const [stepThroughError, setStepThroughError] = useState(null);

  const startStepThrough = () => {
    if (!modelReady || !runningWaveform?.length) return;
    setStepThroughError(null);
    try {
      const steps = getActivationsAtSteps(runningWaveform);
      if (steps?.length) {
        setActivations(steps);
        setStepIndex(0);
        setStepThroughActive(true);
      } else {
        setStepThroughError('Could not compute layer activations.');
      }
    } catch (e) {
      console.error(e);
      setStepThroughError(e?.message ?? 'Step-through failed. Try again.');
    }
  };

  const stepLayerId = activations?.[stepIndex]?.stepId ?? COMPACT_STEPS[stepIndex]?.stepId ?? 'input';
  const stepDesc = LAYER_DESCRIPTIONS[stepLayerId] ?? LAYER_DESCRIPTIONS.input;

  const runAnimation = () => {
    if (running || !diagramRef.current) return;
    setRunning(true);
    setPrediction(null);
    const boxes = d3.select(diagramRef.current).selectAll('.layer');
    const n = layers.length;
    let idx = 0;
    const step = () => {
      boxes.select('rect').attr('opacity', 0.5).attr('stroke', null).attr('stroke-width', null);
      boxes.filter((_, i) => i === idx).select('rect').attr('opacity', 1).attr('stroke', '#0366d6').attr('stroke-width', 2);
      idx++;
      if (idx < n) setTimeout(step, 180);
      else {
        boxes.select('rect').attr('opacity', 1).attr('stroke', null).attr('stroke-width', null);
        if (modelReady && runningWaveform && model === 'compact') {
          try {
            const result = runModelPredict(runningWaveform);
            if (result) {
              setPrediction(
                result.classNames.map((name, i) => ({
                  name,
                  pct: Math.round(result.probabilities[i] * 100),
                  color: CLASS_COLORS[name] ?? '#0366d6',
                  isPredicted: i === result.predictedClass,
                }))
              );
            } else {
              setFallbackPrediction();
            }
          } catch (e) {
            console.error(e);
            setFallbackPrediction();
          }
        } else {
          setFallbackPrediction();
        }
        setRunning(false);
      }
    };

    function setFallbackPrediction() {
      if (sample === 'noise') {
        const rows = [
          { name: 'Noise', pct: 92, color: '#586069', isPredicted: true },
          { name: 'Traffic', pct: 5, color: '#e36209', isPredicted: false },
          { name: 'Earthquake', pct: 3, color: '#d73a49', isPredicted: false },
        ].filter((r) => classNames.includes(r.name));
        setPrediction(rows);
      } else {
        const scale = sample === 'small' ? 1.5 : sample === 'medium' ? 3 : 5;
        const magFactor = Math.min(1, scale / 5);
        let probs = [0.4 * (1 - magFactor) + 0.1, 0.2, 0.4 * magFactor + 0.3];
        const sum = probs.reduce((a, b) => a + b, 0);
        probs = probs.map((p) => (p / sum) * 100);
        const names = ['Noise', 'Traffic', 'Earthquake'];
        setPrediction(
          classNames.map((name, i) => {
            const j = names.indexOf(name);
            const pct = j >= 0 ? Math.round(probs[j]) : 0;
            const isPredicted =
              sample === 'noise' ? name === 'Noise' : name === 'Earthquake' && magFactor > 0.4;
            return {
              name,
              pct,
              color: CLASS_COLORS[name] ?? '#0366d6',
              isPredicted,
            };
          })
        );
      }
    }
    step();
  };

  return (
    <div className="cnn-section">
      <div className="cnn-controls">
        <div className="cnn-controls-row">
          <span>Sample:</span>
          {['noise', 'small', 'medium', 'large'].map((s) => (
            <button key={s} type="button" className={`btn ${sample === s ? 'active' : ''}`} onClick={() => setSample(s)}>
              {s === 'noise' ? 'Noise' : s === 'small' ? 'Small magnitude' : s === 'medium' ? 'Medium magnitude' : 'Large magnitude'}
            </button>
          ))}
        </div>
        <div className="cnn-controls-row">
          <span>Model:</span>
          <button type="button" className={`btn ${model === 'compact' ? 'active' : ''}`} onClick={() => setModel('compact')}>Compact</button>
          <button type="button" className={`btn ${model === 'standard' ? 'active' : ''}`} onClick={() => setModel('standard')}>Standard</button>
        </div>
        <div className="cnn-controls-row cnn-controls-row-actions">
          <button type="button" className="btn btn-primary" onClick={runAnimation} disabled={running}>Run through CNN</button>
          <button type="button" className="btn btn-step" onClick={startStepThrough} disabled={running || !modelReady || !runningWaveform?.length} title="Step through each layer with a visual of the signal at that stage">Step through</button>
        </div>
      </div>
      {stepThroughError && (
        <p className="step-through-error" role="alert">{stepThroughError}</p>
      )}

      {runningWaveform && (
        <div className="running-sample">
          <p className="running-sample-label">Waveform being analyzed: {sampleLabel}</p>
          <div ref={runningRef} className="running-sample-chart" />
          {runningMeta && (
            <div className="running-sample-meta">
              {runningMeta.label === 'Noise' ? (
                <>
                  {runningMeta.station && <span>Station: {runningMeta.station}</span>}
                  {runningMeta.network && <span>Network: {runningMeta.network}</span>}
                </>
              ) : (
                <>
                  {runningMeta.magnitude != null && <span>Magnitude: M{runningMeta.magnitude}</span>}
                  {runningMeta.station && <span>Station: {runningMeta.station}</span>}
                  {runningMeta.network && <span>Network: {runningMeta.network}</span>}
                  {runningMeta.event_time && <span>Time: {runningMeta.event_time}</span>}
                  {runningMeta.event_lat != null && runningMeta.event_lon != null && (
                    <span>Location: {runningMeta.event_lat.toFixed(2)}°N, {runningMeta.event_lon.toFixed(2)}°W</span>
                  )}
                  {runningMeta.distance_km != null && <span>Distance: {runningMeta.distance_km.toFixed(0)} km</span>}
                </>
              )}
            </div>
          )}
        </div>
      )}

      {stepThroughActive && activations?.length > 0 && (
        <div className="step-through-panel">
          <div className="step-through-header">
            <span className="step-through-title">Step {stepIndex + 1} of {activations.length}: {activations[stepIndex].label}</span>
            {model === 'standard' && (
              <span className="step-through-note">(activations from Compact — same layer types)</span>
            )}
            <button type="button" className="close-desc" onClick={() => { setStepThroughActive(false); setActivations(null); setStepThroughError(null); }} aria-label="Close step-through">×</button>
          </div>
          <div className="step-through-controls">
            <button type="button" className="btn" onClick={() => setStepIndex((i) => Math.max(0, i - 1))} disabled={stepIndex === 0}>← Previous</button>
            <button type="button" className="btn" onClick={() => setStepIndex((i) => Math.min(activations.length - 1, i + 1))} disabled={stepIndex === activations.length - 1}>Next →</button>
          </div>
          <p className="step-through-desc">
            <strong>{stepDesc.title}.</strong> {stepDesc.body}
          </p>
          {activations[stepIndex]?.stepId === 'fc' && (
            <div className="step-through-fc-explainer">
              <p className="step-through-desc">
                <strong>How we get from 64 numbers to 2 scores:</strong> The global average pool gave us 64 numbers (one per feature channel). The fully connected layer has a weight matrix <em>W</em> (64×2) and a bias vector <em>b</em> (2). It computes: <em>score<sub>Noise</sub> = W<sub>Noise</sub> · x + b<sub>Noise</sub></em> and <em>score<sub>Earthquake</sub> = W<sub>Earthquake</sub> · x + b<sub>Earthquake</sub></em>. The two bars below are these <strong>raw scores</strong> (before softmax). A higher score means the model leans toward that class; we then turn scores into probabilities with softmax.
              </p>
            </div>
          )}
          <div ref={activationChartRef} className="step-through-chart" />
        </div>
      )}

      <div className="cnn-and-desc">
        <div ref={diagramRef} className="cnn-diagram-wrap" />
        {!stepThroughActive && selectedLayer && LAYER_DESCRIPTIONS[selectedLayer] && (
          <div className="layer-desc-panel">
            <div className="layer-desc-header">
              <h4>{LAYER_DESCRIPTIONS[selectedLayer].title}</h4>
              <button type="button" className="close-desc" onClick={() => setSelectedLayer(null)} aria-label="Close">×</button>
            </div>
            <p>{LAYER_DESCRIPTIONS[selectedLayer].body}</p>
            <p className="layer-desc-hint">Click another layer to see its description.</p>
          </div>
        )}
      </div>

      {prediction && (
        <div className="prediction-panel">
          <div className="prediction-panel-header">
            <h3>Prediction</h3>
            {modelReady && model === 'compact' && (
              <span className="prediction-badge live">Live model</span>
            )}
          </div>
          {!modelReady && (
            <p className="prediction-note">Illustrative scores. Load the compact model to see real predictions.</p>
          )}
          {prediction.map(({ name, pct, color, isPredicted }) => (
            <div key={name} className={`prediction-row ${isPredicted ? 'predicted' : ''}`}>
              <span>{name}</span>
              <div className="prediction-bar">
                <div className="prediction-fill" style={{ width: `${pct}%`, background: color }} />
              </div>
              <span>{pct}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
