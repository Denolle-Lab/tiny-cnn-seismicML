# tiny-cnn-seismicML

Small convolutional neural networks (CNNs) that classify 60 s of vertical ground motion recorded around Anchorage, Alaska, as **Noise, Earthquake, Traffic, Train or Aircraft**. The models have 9k–94k parameters, so they run in a web browser on a classroom Chromebook. They are built for the SeismicML curriculum with the Concord Consortium (CLUE) and with Romig Middle School, which hosts two of the Raspberry Shake seismometers we use.

**Project pages** (GitHub Pages, from [docs/](docs/)):

| Page | What it shows |
|---|---|
| [docs/report.html](docs/report.html) | Illustrated report: the datasets, a map of Anchorage with Romig Middle School, the stations and the earthquakes, a catalog of signal types and of the 550 earthquakes, how the CNN predicts a class, and the limitations |
| [docs/index_draft.html](docs/index_draft.html) | Every trained model: data, grouped split, test and held-out scores, training curves |
| [docs/review/index.html](docs/review/index.html) | Review site: agree or disagree with labeled windows and download your decisions |
| [docs/label_verification_2026-09-14.html](docs/label_verification_2026-09-14.html) | Labeling rules, their parameters, counts and check figures |

## What the project does

1. **Collect** labeled 60 s windows from public seismic data (FDSN): AK network broadbands and strong-motion sensors through EarthScope/IRIS, and Raspberry Shake geophones (network AM) at Anchorage schools.
2. **Label** every window with a rule that can be rerun, never by hand. Earthquakes come from USGS ComCat P picks, traffic from the time of day at school Shakes, trains from the Alaska Railroad timetable, and aircraft from high-frequency bursts seen at two Shakes near ANC runway 15/33.
3. **Train** a compact (9.4k parameter) or standard (94k parameter) 1-D CNN on a subset of classes. The training, validation and test split is grouped by earthquake or by station-day.
4. **Evaluate** on a held-out set of 485 windows recorded after every training date.
5. **Deploy** each model as a `models/<id>/` folder (`metadata.json` + TensorFlow.js `weights.json`) that a browser loads directly.

| Label | Class | Data source | Status |
|---|---|---|---|
| 0 | Noise | quiet hours (01–05 local) at every station; pre-P windows for earthquakes | trained |
| 1 | Traffic | 6 Raspberry Shakes at schools, incl. Romig (AM.R1796, R3130), week of 2026-09-07 | trained |
| 2 | Earthquake | 477 ComCat events, M 3–6, 2020–2025, at AK.FIRE, RC01, SSN | trained |
| 3 | Avalanche | — | planned, [#9](https://github.com/Denolle-Lab/tiny-cnn-seismicML/issues/9) |
| 4 | Train | AK.K222 at Potter Marsh, 100 m from the railroad, May–Sep 2026 | trained |
| 5 | Aircraft | AM.RFD97, R9286 in Turnagain, Aug–Sep 2026 | trained |

## Install

Python 3.10 or newer is recommended. Either environment manager works.

**pip + venv**

```bash
git clone https://github.com/Denolle-Lab/tiny-cnn-seismicML.git
cd tiny-cnn-seismicML
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter ipykernel      # for the notebooks
python -m ipykernel install --user --name=seismic-cnn
```

**conda** (simpler if `usgs-libcomcat` fails to build, because it needs GDAL)

```bash
conda create -n seismic-cnn -c conda-forge python=3.11 fiona obspy pytorch jupyter
conda activate seismic-cnn
pip install -r requirements.txt
```

Check the install:

```bash
python -c "from src.models import CompactSeismicCNN; print(CompactSeismicCNN(num_classes=2, input_channels=1).count_parameters())"
```

The explainer app also needs Node.js 20.19 or newer (see [Deploy](#5-deploy-to-the-browser)).

## Run it

All commands run from the repository root. Waveform files (`*.npy`) are never committed. They are rebuilt from public archives, and `datasets/metadata/` records exactly which windows each set holds.

### 1. Rebuild the training data

```bash
python scripts/collect_ak_events.py --config configs/ak_events.yaml                 # Earthquake + Noise, ~16 min
python scripts/collect_continuous_windows.py --config configs/am_stations.yaml        # Traffic week, school Shakes
python scripts/collect_continuous_windows.py --config configs/train_stations.yaml     # Train, K222 passenger season
python scripts/collect_continuous_windows.py --config configs/aircraft_stations.yaml  # Aircraft, Turnagain, 4 weeks
python scripts/export_metadata.py --check   # every set should print "identical" against datasets/metadata/
```

Output goes to `notebooks/02_labeling/labeled_data/` as `<prefix>_waveforms_<stamp>.npy`, `_labels_`, `_metadata_.csv` and `_summary_.txt`. Add `--max-events 5` to the first command for a quick test. Details of each collector are in [Data collection details](#data-collection-details).

### 2. Train

Open `notebooks/03_training/train_cnn_multiclass.ipynb` and set:

```python
DATA_SOURCE = 'ak'
EXTRA_SOURCES = ['AM_traffic_*', 'AK_train_*', 'AM_aircraft_*']
MODEL_CLASSES_KEY = 'human'       # or 'earthquake', 'rule_based', 'human_traffic', or a list of class names
```

Running the notebook end to end band-passes, crops, z-scores and balances the windows, then makes the grouped split (written to `datasets/splits/`). It trains both architectures and saves the checkpoints, `training_summary_*.txt` and the `models/<id>/` packages. A script version for a config file is:

```bash
python train.py --config configs/compact_config.yaml --save-dir models
```

Class subsets (`src/data/labels.py`, `MODEL_CLASSES`):

| Key | Output classes |
|---|---|
| `earthquake` | Noise / Earthquake |
| `["Noise", "Traffic"]` | Noise / Traffic (a list works too) |
| `rule_based` | Noise / Traffic / Earthquake |
| `human` | Noise / Train / Aircraft |
| `human_traffic` | Noise / Traffic / Train / Aircraft |
| `natural` | Noise / Earthquake / Avalanche (once avalanche data exist) |

### 3. Evaluate on the held-out set

```bash
python scripts/run_heldout_inference.py      # every models/<id> on datasets/heldout_inference/
python scripts/plot_training_curves.py
python scripts/plot_heldout_examples.py
```

Results are written to `datasets/heldout_inference/heldout_predictions.csv`, one row per model and window.

### 4. Predict on any station and time

```bash
jupyter notebook notebooks/04_inference/predict_on_new_station.ipynb
```

Set `NETWORK`, `STATION`, `CHANNEL`, `START_TIME` and `END_TIME`. The notebook downloads the data, slides a 60 s window with 50 % overlap, and plots the class probabilities over time. From the command line:

```bash
python predict.py --model-path models/<checkpoint>.pth --config configs/standard_config.yaml
```

The input must match training: vertical component, 100 Hz, band-pass 2–20 Hz, each 60 s window z-scored.

### 5. Deploy to the browser

A deployable model is a folder `models/<id>/` with `metadata.json` (class names, sampling rate, window length, accuracy) and `weights.json` (TensorFlow.js format). Eleven are in the repo (`compact-v1`…`v6`, `standard-v1`…`v5`). To export a new checkpoint:

```bash
python scripts/export_compact_weights_for_tfjs.py  --model_path models/seismic_cnn_compact_<stamp>.pth  --output_dir /tmp/export
python scripts/export_standard_weights_for_tfjs.py --model_path models/seismic_cnn_standard_<stamp>.pth --output_dir /tmp/export
```

Then follow [docs/generating-model-weights.md](docs/generating-model-weights.md) to assemble the folder. CLUE loads the folder by URL. To run the CNN explainer locally:

```bash
python scripts/export_waveforms_for_explainer.py
python scripts/export_compact_weights_for_tfjs.py
cd explainer-app && npm install && npm run dev     # http://localhost:5173
```

[browser_demo/](browser_demo/) is a plain-HTML demo, and [docs/BROWSER_DEPLOYMENT.md](docs/BROWSER_DEPLOYMENT.md) covers block-coding integration and hosting.

### 6. Rebuild the report page

```bash
python scripts/make_report_page.py    # docs/report.html from scripts/report_template.html + USGS ComCat
```

## The models

Both take `(batch, 1, 6000)`: one vertical channel, 60 s at 100 Hz.

| | `CompactSeismicCNN` | `SeismicCNN` |
|---|---|---|
| Conv blocks | 3 (16, 32, 64 filters; kernels 7, 5, 3) | 4 (32, 64, 128, 128 filters) |
| Head | global average pool → 1 linear layer | global average pool → 2 linear layers, dropout |
| Parameters | ~9.4k | ~94k |
| Held-out accuracy | 92–98 % depending on class subset | 94–98 % |

```python
from src.models import CompactSeismicCNN, SeismicCNN
model = CompactSeismicCNN(num_classes=4, input_channels=1, input_length=6000)
```

Deployed models in `models/` (test = grouped test split, held-out = `datasets/heldout_inference/`; full details on [the model page](docs/index_draft.html)):

| Classes | Compact | Standard |
|---|---|---|
| Noise / Earthquake | `compact-v5`: 99.2 % test, 97.5 % held-out | `standard-v4`: 98.9 % test, 97.5 % held-out |
| Noise / Traffic | `compact-v6`: 95.9 % test, 95.0 % held-out | `standard-v5`: 97.4 % test, 94.0 % held-out |
| Noise / Traffic / Earthquake | `compact-v3`: 94.5 % test, 95.3 % held-out | `standard-v2`: 96.0 % test, 95.7 % held-out |
| Noise / Traffic / Train / Aircraft | `compact-v4`: 94.3 % test, 92.5 % held-out | `standard-v3`: 96.1 % test, 95.1 % held-out |
| Noise / Earthquake, July 2026 (earlier split) | `compact-v2`, `compact-v1` | `standard-v1` |

Each conv block is convolution → batch norm → ReLU → max-pool. A last-layer feature sees about 0.45 s of signal (compact model), and the global average pool reports how much of each learned pattern occurs in the minute. [docs/report.html](docs/report.html#cnn) explains this step by step.

## Limitations

- **Weak labels.** Labels come from rules (time of day, timetable, catalog), so some windows are mislabeled. Use the review site to flag them.
- **One site per class.** Trains come only from K222, aircraft only from Turnagain, and earthquakes only from AK broadbands. A model may partly learn the sensor or the site. The held-out set uses the same stations, so it cannot rule this out.
- **No amplitude, short memory.** Per-window z-scoring removes loudness, and global pooling removes timing within the window.
- **Closed set.** Every window is forced into one known class. There is no "unknown" output and no handling of mixed windows.
- **Limited coverage.** M ≥ 3 only, one week of traffic in September, no winter data, no avalanches yet. Held-out Train windows come from a test split, not from new days.

## Repository layout

```
src/            models (cnn.py), labels and class subsets, collectors (data/collect.py), training utilities
scripts/        collectors, export, held-out inference, plotting, report and review-site builders
configs/        collection configs (ak_events, am/train/aircraft_stations, arr_schedule) and training configs
datasets/       committed metadata: per-set window lists, split manifests, held-out set and predictions
models/         deployable models/<id>/ folders and training summaries
notebooks/      01_data_exploration, 02_labeling, 03_training, 04_inference
docs/           GitHub Pages site, figures, station tables, deployment guides
explainer-app/  React + TensorFlow.js CNN explainer
browser_demo/   minimal browser classifier
```

See [ORGANIZATION.md](ORGANIZATION.md) and [notebooks/README.md](notebooks/README.md) for more.

## Data collection details

**Earthquakes** (`scripts/collect_ak_events.py`, `configs/ak_events.yaml`). The collector queries USGS for M 3–7 events within 150 km of Anchorage (2020-01-01 to 2025-12-01) and takes the reviewed P picks from ComCat (`usgs-libcomcat`). It downloads vertical broadband data from IRIS and cuts a 120 s earthquake window (P at 30 s) plus a 60 s pre-event noise window per station, band-passed 2–20 Hz at 100 Hz. The 98 noise windows rejected by eye in July 2026 are dropped (`docs/ak_dropped_windows.csv`). Command-line flags override the config. `--zscore` reproduces the July 2026 `AK_*` files byte for byte.

**Continuous classes** (`scripts/collect_continuous_windows.py`). This collector writes 60 s / 100 Hz windows in the same layout with a provisional label, per-window features and blank `reviewed` / `review_label` columns. `--review-sheet N` adds a PNG grid and a CSV for review by eye. Label modes:

- `--label-from timeofday`: daytime = Traffic, 01–05 local = Noise. A station-day is kept only if its daytime 5–30 Hz level is at least 2× its night level.
- `--label-from events --events file.csv`: windows around listed times. The Train set uses the Alaska Railroad timetable (`configs/arr_schedule.yaml` → `scripts/make_train_events.py` → `configs/events/arr_summer_2026.csv`), searched ±15 min and kept where the 5–30 Hz rms stands out.
- `--label-from rule`: rms threshold against a quiet reference. The Aircraft set uses this with a 20–45 Hz band on both Turnagain Shakes.

```bash
python scripts/collect_continuous_windows.py --network AM --station R4017 \
    --start 2026-09-09 --end 2026-09-10 --class-name Traffic --label-from timeofday --review-sheet 24
```

Station-to-school matches are in `docs/am_station_school_matches.csv` and distances to the railroad and runways in `docs/station_rail_runway_distances.csv`. ADS-B flight times for aircraft can be fetched with `scripts/fetch_flights_opensky.py` (needs an OpenSky account). The shared FDSN client, preprocessing chain and file layout live in `src/data/collect.py`, which imports without torch.

**Label scheme.** `src/data/labels.py` (`LABEL_MAP`) is append-only: integers on disk never change. `select_classes(X, y, key)` keeps one subset and remaps it to outputs 0…K−1, with Noise always at 0.

## Citation

```bibtex
@software{tiny-cnn-seismicML,
  title  = {tiny-cnn-seismicML: Lightweight CNNs for Seismic Signal Classification in Anchorage},
  author = {Denolle Lab},
  year   = {2026},
  url    = {https://github.com/Denolle-Lab/tiny-cnn-seismicML}
}
```

## License and contact

See [LICENSE](LICENSE). Questions and contributions: open an issue or a pull request on GitHub.
