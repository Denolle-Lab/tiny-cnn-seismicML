import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import * as d3 from 'd3';
import { useWaveformData } from '../hooks/useWaveformData';
import './SingleEventView.css';

const SR = 100;

function FitBounds({ stations, event }) {
  const map = useMap();
  useEffect(() => {
    if (!event || !stations?.length) return;
    const bounds = [[event.lat, event.lon], [event.lat, event.lon]];
    stations.forEach((s) => bounds.push([s.lat, s.lon]));
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 12 });
  }, [map, event, stations]);
  return null;
}

function drawStationWaveform(container, data) {
  if (!container || !data?.length) return;
  d3.select(container).selectAll('*').remove();
  const len = data.length;
  const timeMax = (len - 1) / SR;
  const margin = { top: 10, right: 12, bottom: 28, left: 44 };
  const width = Math.max(300, container.clientWidth - margin.left - margin.right);
  const height = 140 - margin.top - margin.bottom;
  const xScale = d3.scaleLinear().domain([0, timeMax]).range([0, width]);
  const yExtent = d3.extent(data);
  const pad = Math.max(0.5, (yExtent[1] - yExtent[0]) * 0.15) || 0.5;
  const yScale = d3.scaleLinear().domain([yExtent[0] - pad, yExtent[1] + pad]).range([height, 0]).nice();
  const line = d3.line().x((d, i) => xScale(i / SR)).y((d) => yScale(d)).curve(d3.curveMonotoneX);
  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('height', 140)
    .attr('viewBox', `0 0 ${width + margin.left + margin.right} 140`);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  g.append('path').attr('d', line(data)).attr('fill', 'none').attr('stroke', '#d73a49').attr('stroke-width', 1.2).attr('stroke-linejoin', 'round');
  g.append('g').attr('class', 'axis axis-x').attr('transform', `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
  g.append('g').attr('class', 'axis axis-y').call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0));
  svg.append('text').attr('class', 'axis-label').attr('x', margin.left + width / 2).attr('y', 136).attr('text-anchor', 'middle').text('Time (s)');
}

export function SingleEventView() {
  const { data, error } = useWaveformData();
  const [selectedStation, setSelectedStation] = useState(null);
  const waveformRef = useRef(null);

  const singleEvent = data?.single_event ?? null;
  const event = singleEvent?.event ?? null;
  const stations = singleEvent?.stations ?? [];

  useEffect(() => {
    if (selectedStation != null && stations[selectedStation] && waveformRef.current) {
      drawStationWaveform(waveformRef.current, stations[selectedStation].waveform);
    }
  }, [selectedStation, stations]);

  if (error) return <p className="single-event-error">Could not load single-event data. Run the export script.</p>;
  if (!singleEvent) return <p className="single-event-loading">Loading single event…</p>;

  return (
    <div className="single-event-view">
      <div className="single-event-header">
        <h2>Single earthquake event</h2>
        <p className="event-info">
          <strong>M{event.magnitude}</strong> · {event.event_time} · {event.lat.toFixed(2)}°N, {event.lon.toFixed(2)}°W
        </p>
        <p className="event-hint">Click a station on the map to see the waveform recorded there.</p>
      </div>

      <div className="insight-panel">
        <h3>What can we learn from this map?</h3>
        <ul>
          <li><strong>Distance</strong> — Stations farther from the epicenter see the waves arrive later and the signal is usually weaker. Compare waveforms: the closest station often has the sharpest onset; the farthest may be noisier.</li>
          <li><strong>Topography & geology</strong> — This map shows terrain. Seismic waves can be amplified in soft sediments and basins and damped by hard rock. Stations in different geologic settings can record different shaking even at similar distances.</li>
          <li><strong>Network coverage</strong> — Multiple stations around an event let seismologists locate the earthquake and estimate magnitude. Dense networks give better resolution.</li>
        </ul>
      </div>

      <div className="map-wrap">
        <MapContainer center={[event.lat, event.lon]} zoom={8} className="single-event-map" scrollWheelZoom={true}>
          <TileLayer
            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="https://opentopomap.org">OpenTopoMap</a>'
            maxZoom={17}
          />
          <FitBounds stations={stations} event={event} />
          <Marker position={[event.lat, event.lon]}><Popup>Earthquake</Popup></Marker>
          {stations.map((st, i) => (
            <Marker key={st.station} position={[st.lat, st.lon]} eventHandlers={{ click: () => setSelectedStation(i) }}>
              <Popup>{st.station} ({st.distance_km.toFixed(0)} km)</Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {selectedStation != null && stations[selectedStation] ? (
        <div className="station-waveform-panel">
          <h3>
            {stations[selectedStation].station} ({stations[selectedStation].network}) — {stations[selectedStation].distance_km.toFixed(0)} km from event
          </h3>
          <div ref={waveformRef} className="station-waveform-chart" />
        </div>
      ) : (
        <p className="station-prompt">Click a station marker to view its waveform.</p>
      )}
    </div>
  );
}
