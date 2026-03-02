import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useWaveformData } from '../hooks/useWaveformData';
import './WaveformSection.css';

const SR = 100;

function drawChart(container, data, color) {
  if (!container || !data?.length) return;
  d3.select(container).selectAll('*').remove();
  const len = data.length;
  const timeMax = (len - 1) / SR;
  const margin = { top: 8, right: 12, bottom: 28, left: 44 };
  const width = Math.max(200, container.clientWidth - margin.left - margin.right);
  const height = 160 - margin.top - margin.bottom;
  const xScale = d3.scaleLinear().domain([0, timeMax]).range([0, width]);
  const yExtent = d3.extent(data);
  const yScale = d3.scaleLinear().domain(yExtent).range([height, 0]).nice();
  const line = d3.line().x((d, i) => xScale(i / SR)).y((d) => yScale(d)).curve(d3.curveMonotoneX);
  const svg = d3.select(container)
    .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', 160)
    .attr('viewBox', `0 0 ${width + margin.left + margin.right} 160`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  g.append('path').attr('d', line(data)).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5).attr('stroke-linejoin', 'round');
  g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
  g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0));
  svg.append('text').attr('class', 'axis-label').attr('x', margin.left + width / 2).attr('y', 158).attr('text-anchor', 'middle').text('Time (s)');
}

export function WaveformSection() {
  const chartRef = useRef(null);
  const { data, error } = useWaveformData();
  const [mag, setMag] = useState('small');

  const eqRecord = data?.earthquake?.[mag]?.[0] ?? data?.earthquake?.small?.[0] ?? data?.earthquake?.medium?.[0] ?? data?.earthquake?.large?.[0];
  const waveform = eqRecord?.waveform ?? null;

  useEffect(() => {
    if (waveform && chartRef.current) drawChart(chartRef.current, waveform, '#d73a49');
  }, [waveform, mag]);

  if (error) {
    return (
      <p className="waveform-error">
        Could not load waveforms. Run <code>python scripts/export_waveforms_for_explainer.py</code> from repo root.
      </p>
    );
  }

  if (!data) return <p className="waveform-loading">Loading waveforms…</p>;

  return (
    <div className="waveform-section">
      <div className="waveform-card">
        <h3>🌋 Earthquake</h3>
        <div className="magnitude-tabs">
          {['small', 'medium', 'large'].map((m) => (
            <button
              key={m}
              type="button"
              className={`tab ${mag === m ? 'active' : ''}`}
              onClick={() => setMag(m)}
            >
              {m === 'small' ? 'Small' : m === 'medium' ? 'Medium' : 'Large'}
            </button>
          ))}
        </div>
        <div ref={chartRef} className="waveform-chart" role="img" aria-label="Earthquake waveform" />
        <p className="waveform-hint">Sampling rate: 100 Hz. Hover to see P-wave and S-wave regions.</p>
      </div>
    </div>
  );
}
