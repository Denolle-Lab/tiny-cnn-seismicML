# compact-v1

Deployed seismic classification model consumed by the CLUE Wave Runner tile.

- `metadata.json` — describes the model (architecture, classes, sample rate, etc.) for CLUE.
- `weights.json` — model weights in TensorFlow.js JSON format.

The `metadata.json` `$schema` field points at the format definition (CLUE's
`ModelMetadata`); models consumed by CLUE must follow it.

## Provenance

These weights are the TF.js export of Derek's trained compact CNN; the file is
byte-identical to `compact_weights.json` from his explainer app
(`yaoderek/tiny-cnn-seismicML`). It is a real trained model (3 conv blocks,
classes: Noise, Earthquake), not a placeholder.

## Creating another model

To add a new model (e.g. `compact-v2`), follow
[docs/generating-model-weights.md](../../docs/generating-model-weights.md):
train → export to TF.js JSON → assemble this folder → use. Bump the `id`
(and folder name) when retraining so existing CLUE event paths keyed on the old
id stay valid.
