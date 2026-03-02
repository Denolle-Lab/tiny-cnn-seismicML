# Earthquake CNN Explainer (React)

A React frontend that explains how the project’s CNN classifies seismic waveforms (Noise vs Earthquake). Inspired by [CNN Explainer](https://poloclub.github.io/cnn-explainer/) (Polo Club), adapted for 1D seismic data. Features a **live compact model** in the browser (TensorFlow.js), step-through layer visualization, and results on many waveforms.

## Setup (from repo root)

1. **Export waveform data** (requires labeled data in `notebooks/02_labeling/labeled_data/`):

   ```bash
   python scripts/export_waveforms_for_explainer.py
   ```

   Writes `explainer-app/public/waveforms.json`.

2. **Export compact model weights** (requires a trained compact `.pth` in `models/`):

   ```bash
   python scripts/export_compact_weights_for_tfjs.py
   ```

   Writes `explainer-app/public/models/compact_weights.json`. Default path: `models/seismic_cnn_compact_ak_20260219_102948.pth`. Override with `--model_path`.

3. **Install and run**:

   ```bash
   cd explainer-app
   npm install
   npm run dev
   ```

   Open http://localhost:5173

## Build

```bash
npm run build
```

Output is in `dist/`. Test with `npm run preview` or deploy to any static host.

## Stack

- **React** + **Vite**
- **TensorFlow.js** for the compact CNN in the browser
- **D3.js** for waveform charts and CNN layer diagram
- **Leaflet** + **react-leaflet** for the Single Event map (OpenTopoMap)

## Tabs

- **Explainer** — What is a CNN (glossary), earthquake waveforms (Noise / Small–Large magnitude), CNN section with sample + model choice, **Run through CNN** (live predictions when weights are loaded), **Step through** (layer-by-layer with activation visuals), layer diagram (expand on click/step), results section with test-prediction grid and training curves, hyperparameters.
- **Single event** — One earthquake, insight panel, map with stations; click a station to see its waveform.
