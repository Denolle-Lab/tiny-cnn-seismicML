# Generating model weights

A deployable model is a folder `models/<model-id>/` containing two files:

- `metadata.json` — describes the model for the consumer (CLUE).
- `weights.json` — the model's weights in TensorFlow.js JSON format, exported
  from a trained PyTorch checkpoint.

This guide covers producing those files. `compact-v1` was created this way (see
[models/compact-v1/README.md](../models/compact-v1/README.md) for its provenance).

## 1. Train a model

Run the training notebook to produce a PyTorch checkpoint:

- `notebooks/03_training/train_cnn_multiclass.ipynb` (compact architecture)

This writes a checkpoint such as `models/seismic_cnn_compact_<...>.pth`.

## 2. Export to TensorFlow.js JSON

Convert the checkpoint to the TF.js weight format with the export script:

```bash
python scripts/export_compact_weights_for_tfjs.py \
  --model_path models/seismic_cnn_compact_<...>.pth \
  --output_dir /tmp/export
```

This writes `/tmp/export/compact_weights.json` in the same JSON format used by
`weights.json`. The architecture in the script must match the trained model —
`export_compact_weights_for_tfjs.py` targets the compact CNN; add a sibling
exporter for other architectures.

## 3. Assemble the model folder

```bash
mkdir -p models/<model-id>
cp /tmp/export/compact_weights.json models/<model-id>/weights.json
# then create models/<model-id>/metadata.json
```

Use [models/compact-v1/metadata.json](../models/compact-v1/metadata.json) as the
template for the fields; the metadata format is defined by the consuming
application (CLUE's `ModelMetadata`, referenced via the `$schema` URL in that file).
Bump `id` (e.g. `compact-v2`) when retraining so events keyed on the old id remain
valid.

## 4. Use the model

A model folder in this format is loaded and run in the browser by the consuming
application. For the CLUE Wave Runner tile, the folder is deployed to a CDN and
fetched at runtime. Steps 1–3 (train → export → assemble) are all that is needed to
produce a model in this format.

## Using a different architecture

The `architecture` field in `metadata.json` is a free-form string in the schema, but
the consuming application maps it to a build function at runtime. CLUE keeps a
registry (`ARCHITECTURES` in `shared/seismic/seismic-architectures.ts`) — currently
`compact` and `standard` (plus `placeholder` for testing). A model whose
`architecture` is not in the registry fails to load with an `Unknown architecture`
error.

- **Reusing `compact` or `standard`:** just set the field — no application changes are
  needed — and use the matching exporter (`export_compact_weights_for_tfjs.py` for
  `compact`).
- **A genuinely new architecture** requires three pieces that must agree:
  1. A TF.js build function added and registered in CLUE's `ARCHITECTURES` map,
     mirroring the trained model's layers.
  2. An exporter that writes `weights.json` with layer names matching that build
     function — weights are loaded by name. Copy `export_compact_weights_for_tfjs.py`
     as the pattern for a new architecture.
  3. The PyTorch training code that produces the new model.

See CLUE's [ML model integration design](https://github.com/concord-consortium/collaborative-learning/blob/master/docs/seismic/ml-model-integration-design.md)
for how architectures are registered and consumed.
