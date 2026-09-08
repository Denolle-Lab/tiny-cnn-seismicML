# Multi-class pipeline (scripts)

Command-line path from verified catalog events to a CLUE model folder,
for any number of classes. The rationale and class choices are in
[multiclass-model-plan.md](multiclass-model-plan.md); this page is the
how-to.

```
inventory_event_types.py     what exists in ComCat for the region
        |
build_dataset.py  (one run per class, per network)
        |   notebooks/02_labeling/labeled_data/runs/<class>_<net>_<ts>/
        v
review_windows.ipynb         page through, write exclude.txt
        |
merge_datasets.py            relabel by name, cap classes, drop excluded
        |   notebooks/02_labeling/labeled_data/dataset_<n>class_<ts>/
        v
train_multiclass.py          event-level split, compact + standard
        |   models/seismic_cnn_<arch>_<tag>_<ts>.pth + metrics
        v
package_model.py             models/<model-id>/{metadata,weights}.json + README
```

All scripts run from the repo root. Class names and their ComCat event
types live in `configs/classes.json`; index 0 is always `Noise`.

## 0. Inventory

```bash
python scripts/inventory_event_types.py
python scripts/inventory_event_types.py --radius-km 400 --types "ice quake,landslide,avalanche"
```

Prints event counts per type and magnitude floor. Decide the class list
from this before downloading anything.

## 1. Build one run per class

```bash
python scripts/build_dataset.py --class Earthquake --network AK --max-events 400
python scripts/build_dataset.py --class Blast      --network AK
python scripts/build_dataset.py --class "Ice quake" --network AK --radius-km 300
python scripts/build_dataset.py --class Earthquake --network AM --max-events 200   # Raspberry Shake
```

What a run does:

- queries ComCat for the class's `eventtype` values inside the region box
  (`--radius-km` around Anchorage by default), magnitude floor from
  `classes.json` or `--min-magnitude`;
- finds stations with a vertical channel (IRIS for AK, the Raspberry
  Shake server for AM), takes the nearest `--max-stations-per-event`
  within `--max-station-km`;
- resolves the onset: ComCat P pick when the event has phase data, else a
  TauP (iasp91) travel time, refined with STA/LTA (`--onset`, `--no-stalta`);
- downloads one trace, bandpasses 2-20 Hz, resamples to 100 Hz, cuts the
  event window `[onset-30 s, onset+90 s]` and a noise window
  `[onset-90 s, onset-30 s]`, z-normalizes each;
- writes `waveforms.npy` (ragged object array), `labels.npy`,
  `metadata.csv` (event_id, station, network, magnitude, distance,
  onset_source, window_type, ...), `classes.json`, `run_config.json`.

Every event window brings its own noise window, so the Noise class always
has the same station and instrument mix as the event classes.

## 2. Review

Open `notebooks/02_labeling/review_windows.ipynb`, point it at the run,
page through, list bad indexes, run the last cell to write
`<run>/exclude.txt`. Also look at the median spectrum per class: if two
classes have the same spectrum, the CNN will not separate them either.

## 3. Merge

```bash
python scripts/merge_datasets.py notebooks/02_labeling/labeled_data/runs/*_AK_* \
    notebooks/02_labeling/labeled_data/runs/earthquake_AM_* \
    --cap Earthquake=1500 --out notebooks/02_labeling/labeled_data/dataset_4class
```

Relabels by class name against `configs/classes.json`, applies
`exclude.txt`, caps classes by dropping whole events (so the split stays
clean), and prints per-class window and event counts. `--classes` overrides
the class list if you want to drop one.

## 4. Train

```bash
python scripts/train_multiclass.py --dataset notebooks/02_labeling/labeled_data/dataset_4class
python scripts/train_multiclass.py --dataset ... --models standard --epochs 80 --classes Noise,Earthquake,Blast
```

- Splits 70/15/15 **by event id** before cropping. Crops of one window, and
  recordings of one event at several stations, never straddle a split.
- Expands each 120 s event window into one onset-containing crop plus one
  free crop for training (one crop for val/test); noise windows pass
  through.
- Inverse-frequency class weights, light augmentation (noise, shift,
  amplitude), Adam, ReduceLROnPlateau, gradient clipping, early stopping
  on validation loss.
- Reports per-class recall and a confusion matrix on the test set, plus
  accuracy per network (AK vs AM) so the shake number is visible.
- Saves `models/seismic_cnn_<arch>_<tag>_<ts>.pth` with `class_names`,
  `architecture`, `input_length`, `sampling_rate`, preprocessing and the
  metrics inside the checkpoint, plus `metrics_*.json`,
  `training_summary_*.txt` and `confusion_*.png`.

Acceptance bar before packaging: per-class recall at or above 0.80 on the
test set, and at or above 0.70 on AM windows if they are present. The
script prints OK/LOW per class at the end.

## 5. Package

```bash
python scripts/package_model.py --checkpoint models/seismic_cnn_standard_dataset_4class_<ts>.pth \
    --model-id standard-v2 --instrument-types H
```

Creates `models/standard-v2/` with `weights.json` (TF.js layout for the
`compact` or `standard` build function in CLUE), `metadata.json` and a
README carrying the test confusion matrix. Bump the model id for every
retrain; CLUE keys stored events on it.

## Testing without network access

```bash
python tests/test_pipeline.py
```

Synthesizes four classes, runs cut -> merge -> train -> package end to
end, and checks the metadata and weight keys. Needs numpy, scipy,
pandas, scikit-learn and torch; obspy is only needed for the catalog side.
