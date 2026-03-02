# Explainer App Improvement Plan

## Goals

1. **Run a real CNN in the browser** using TensorFlow.js (or equivalent), with real inference and **post-run visualizations** (e.g. class probabilities, optional activation view).
2. **Make the page sleek and responsive** so it feels like one cohesive experience, not four blocks thrown together.

---

## Part A: Real CNN in the Browser

### A1. Model choice and format

- **Model**: Use the trained **Compact** CNN (1 channel, 6000 samples, 3 classes: Noise, Traffic, Earthquake).
- **Path**: Implement the same architecture in TensorFlow.js and load weights exported from the PyTorch checkpoint. This avoids ONNX conversion issues and keeps the bundle small.
- **Fallback**: If the weight file is missing, show a clear message and keep the “Run through CNN” animation; optionally show a “Demo mode” with placeholder scores.

### A2. Preprocessing in the browser

- Match training: **detrend** (subtract mean), **normalize** (zero mean, unit variance). Optionally a simple high-pass approximation if the training pipeline used it; otherwise keep it minimal for reliability.
- Input shape: `[1, 1, 6000]` (batch, channels, time).

### A3. Post-run value

- **Real prediction**: Replace fake “illustrative” scores with **actual softmax probabilities** from the loaded model.
- **Visualizations**:
  - **Class probability chart**: Bar chart (or horizontal bars) of the three class scores with clear labels and a highlight for the predicted class.
  - **Optional**: First-layer activation strip (run input through the first conv + ReLU and plot the 1D activation). Adds educational value and “wow” factor.

### A4. Technical tasks

- Add script to export Compact PyTorch weights to a JSON format that the TF.js model can load (with correct transposes for conv kernels).
- In the app: TF.js Layers model that mirrors `CompactSeismicCNN`, load weights, run `model.predict()`, then render probability chart and optional activation viz.
- Only support Compact in the browser (Standard can remain as diagram-only) to keep the page fast and simple.

---

## Part B: UI/UX Overhaul

### B1. Layout and structure

- **Single scroll experience**: One main column with clear visual hierarchy. Sections are distinct but part of one flow (hero → intro → waveforms → CNN → layer explainer → footer).
- **Max width and spacing**: Constrain content width (e.g. 720–900px for body text), center the column, use consistent vertical rhythm (e.g. 32–48px between sections).
- **Cards/sections**: Each major block (waveforms, CNN, single event) in a subtle card: light border or shadow, rounded corners, consistent padding. No “four boxes on a white background” look.

### B2. Typography and hierarchy

- **Font**: Keep a readable system stack or add one distinctive font for headings (e.g. one Google Font) so the page doesn’t look generic.
- **Scale**: Clear heading levels (h1 → h2 → h3), consistent line-height and margin so copy doesn’t feel cramped.
- **Contrast**: Body text and muted text (e.g. captions, metadata) clearly differentiated.

### B3. Color and polish

- **Palette**: Refined, consistent (e.g. one primary accent for CTAs and links, neutral grays for text and borders). Optional: very subtle gradient or dark header for a more “product” feel.
- **Buttons**: Primary action (e.g. “Run through CNN”) clearly stands out; secondary options (sample/model toggles) visually secondary.
- **Charts**: Axis labels, grid or tick lines where helpful, colors that match the palette (e.g. same class colors in waveform and prediction chart).

### B4. Responsiveness

- **Breakpoints**: Mobile (stack controls, full-width charts, larger tap targets), tablet, desktop.
- **CNN section**: On small screens, layer diagram stacks or scrolls horizontally; controls wrap; prediction chart remains readable.
- **Single event**: Map and station list/waveforms stack on small screens.

### B5. Cohesion

- **Navigation**: Tabs (Explainer / Single event) styled as part of the header; active state and hover clear.
- **Footer**: Compact, consistent with the rest of the style.
- **Glossary**: Modal or inline styling aligned with the new look.

---

## Part C: Delivered Value

- **Educational**: User sees a real waveform, runs a real (lightweight) CNN, and gets real class probabilities plus an optional peek at what the first layer “sees.”
- **Trust**: No fake numbers; when the model is loaded, the page clearly indicates “Live model” or similar.
- **Usability**: One smooth, readable, responsive page that works on phone and desktop and feels intentional rather than thrown together.

---

## Implementation order

1. Export Compact weights and add TF.js model + loader in the app; wire “Run through CNN” to real inference and real probability display.
2. Add post-run visualization (probability chart, then optional activation strip if feasible).
3. Apply layout, typography, color, and responsiveness (Part B) so the Explainer and Single event tabs both use the new system.
