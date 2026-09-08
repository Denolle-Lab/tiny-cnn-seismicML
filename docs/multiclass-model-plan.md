# Plan: four-class models for the SeismicML classroom deployment

Status: proposal, September 2026. Owner: Marine Denolle. No student is
assigned; the plan is sized for one person working part time with the
existing pipeline, and it names the fallback at each step.

## 1. Where we are

Shipped models, all two-class (Noise, Earthquake), all in the CLUE
`models/<id>/{metadata.json,weights.json}` format:

| id            | arch     | trained on                    | test acc | notes                          |
|---------------|----------|-------------------------------|----------|--------------------------------|
| compact-v1    | compact  | AM (Raspberry Shake) catalog  | ~87%     | Derek Yao, Feb 2026            |
| compact-v2    | compact  | AK broadband, 60 s windows    | 99.3%    | Alex Rose, Jul 2026            |
| standard-v1   | standard | AK broadband, 60 s windows    | 99.3%    | Alex Rose, Jul 2026            |

The partner request (Concord Consortium, Aug 2026) is two more classes,
one natural and one human-made, for a classroom run starting Nov 1.

Three things in the current pipeline must change before any retrain,
independent of which classes we pick:

1. **Leaky split.** `notebooks/03_training/train_cnn_multiclass.ipynb`
   makes two random crops per earthquake window (cell 8) and then does a
   random `train_test_split` over the crops (cell 9). Crops from the same
   source window, and windows from the same event recorded at nearby
   stations, land on both sides of the split. The 99.3% numbers are not a
   deployment estimate. Split by event id (the `AK_metadata_*.csv` has
   one) before cropping. The SeisBench save cell already does an
   event-level split; reuse that logic.
2. **Hardcoded class map.** Cell 9 assumes labels come from
   `{0: Noise, 1: Traffic, 2: Earthquake}` and only remaps when exactly
   two labels are present. Replace with a class list read from the
   dataset (a `classes.json` sidecar written by the labeling script), so
   any number of classes works without editing the notebook.
3. **Metadata instrument type.** compact-v2 and standard-v1 declare
   `"instrument_types": ["B"]`. In SEED channel codes the second letter
   is the instrument code and B is a band code, not an instrument. AK
   BHZ/HHZ and Raspberry Shake EHZ are all instrument code H. CLUE's
   design doc says it filters on H and L. Confirm with Scott Cytacki,
   then ship `["H"]` (or `["H","L"]` as compact-v1 does).

Training data is gitignored, so the July dataset lives only on the
machine that built it. The plan below regenerates everything from
catalogs so it is reproducible.

## 2. Class choice: what can actually be labeled in three weeks

A class is trainable only if a few hundred verified onsets exist with
station coverage. Ranked by feasibility:

**Human-made**

- *Quarry blast / explosion.* USGS ComCat carries `eventtype=quarry
  blast` and `explosion`, and the Alaska Earthquake Center labels them
  routinely (Usibelli mine near Healy, road and construction blasts).
  They have P picks, so the existing P-window logic applies unchanged.
  Waveforms are impulsive, high-frequency, shallow, daytime: a good
  teaching contrast with earthquakes. **Recommended.**
- *Airplanes, snowplows, foot traffic.* No catalog. Airplanes would need
  ADS-B history matched to a shake near ANC or Merrill Field, and the
  ground-motion signal is weak. Not for the November deadline. Chris
  already found human-caused signals hard to isolate on the school
  shakes (Feb 2026).

**Natural, non-earthquake**

- *Ice quake / glacial event.* ComCat `eventtype=ice quake`; AEC labels
  some. Count unknown until queried (see step 3.1). If a few hundred
  exist since 2015, this is the first choice.
- *Landslide / avalanche.* ComCat has `landslide` and `avalanche` types
  but Alaska coverage is sparse. The IRIS Exotic Seismic Events Catalog
  (ESEC) adds a few hundred mass-movement events worldwide with station
  lists; onsets are emergent and need STA/LTA rather than a pick.
  Usable as a supplement, not a primary source.
- *Volcanic seismicity.* AVO network (AV) events during the 2024-2025
  Spurr unrest are abundant and catalogued. They are physically
  earthquakes, so a waveform CNN may not separate them from tectonic
  events; volcanic tremor is distinct but rarer. Honest pedagogically
  only if framed as "volcanic vs tectonic source".
- *Distant (teleseismic) earthquake.* Abundant, catalogued, clearly
  natural, and the waveform is unmistakably different (emergent,
  long-period, minutes long). Not a non-earthquake class, but it is the
  safest fallback if the ice quake count is too low.

Recommendation to bring to the Concord meeting: **Noise / Earthquake /
Blast / Ice quake**, with **Distant earthquake** as the documented
fallback for the fourth slot. Decide the fallback trigger up front: if
fewer than ~150 ice-quake events with usable waveforms exist, switch.

## 3. Data curation

### 3.1 Inventory (day 1)

Query counts before promising anything. Alaska box, 2015 to present:

```python
from obspy.clients.fdsn import Client
usgs = Client("USGS")
box = dict(minlatitude=55, maxlatitude=66, minlongitude=-160, maxlongitude=-140,
           starttime="2015-01-01")
for et in ["earthquake", "quarry blast", "explosion", "ice quake",
           "landslide", "avalanche", "volcanic eruption", "other event"]:
    try:
        n = len(usgs.get_events(eventtype=et, **box, minmagnitude=1.5))
    except Exception as e:
        n = f"error: {e}"
    print(f"{et:20s} {n}")
```

Record the counts in this file. They decide the class list.

### 3.2 One labeling script instead of per-class notebooks

Refactor `notebooks/02_labeling/download_AK_only_data.ipynb` into
`scripts/build_dataset.py` with a `--class` argument that maps a ComCat
`eventtype` to a label. Keep everything that works today:

- 100 Hz, vertical component, bandpass 2-20 Hz, per-window
  z-normalization.
- Onset from ComCat P picks (libcomcat) when present; fall back to a
  1D velocity estimate plus STA/LTA refinement (already in the notebook)
  for emergent sources.
- Event window `[P-30 s, P+90 s]` so the trainer can random-crop 60 s
  segments; noise window `[P-90 s, P-30 s]` from the same trace.
- Write `AK_waveforms_*.npy`, `AK_labels_*.npy`, `AK_metadata_*.csv`
  and a new `classes.json` (`["Noise","Earthquake","Blast","Ice quake"]`).

Per-class settings:

| class      | ComCat eventtype        | onset        | stations       | target count |
|------------|-------------------------|--------------|----------------|--------------|
| Earthquake | earthquake, M>=2.5      | P pick       | AK, <150 km    | ~1500 windows (cap) |
| Blast      | quarry blast, explosion | P pick       | AK, <150 km    | as many as exist, aim >=400 |
| Ice quake  | ice quake               | pick or STA/LTA | AK, <200 km | aim >=300 |
| Noise      | pre-P of every event above | n/a       | same           | ~1 per event window |

Cap the earthquake class so it does not dominate; the trainer's
inverse-frequency class weights handle the rest.

### 3.3 Close the instrument gap

The classroom runs on Raspberry Shake EHZ (4.5 Hz geophone), the July
models were trained on broadband. Two cheap steps:

- Re-run the same script against the AM network (the
  `download_earthquake_catalog.ipynb` path) for Earthquake and Noise on
  shakes within 150 km of Anchorage, and include those windows in
  training. Blasts and ice quakes will mostly be AK-only; that is fine
  as long as Noise and Earthquake carry shake examples.
- Hold out a shake-only test set. Report per-class recall on it
  separately from the AK test set. This is the number Concord should
  see, not the AK number.

### 3.4 Manual review

The notebook already has a paging window viewer and a
`remove_windows(bad_indices)` step. Budget one afternoon per new class
to page through it and drop mislabeled or empty windows. For ice quakes
this is not optional; catalog labels are less reliable than for blasts.

## 4. Training

- Event-level split 70/15/15 using the event id column; crop after
  splitting.
- Train compact (9k params) and standard (94k params) as now. Expect
  the compact model to struggle at four classes; standard is the likely
  deliverable. Both architectures are already registered in CLUE, so
  neither needs Concord dev time.
- Only if standard fails the acceptance bar below: a wider standard
  variant (`conv4` 128 -> 256, `fc1` 64 -> 128, roughly 250k params).
  That is a new architecture and needs a TF.js build function in CLUE's
  registry plus a matching exporter (see
  `docs/generating-model-weights.md`). Raise this at the meeting so the
  dev time is known before it is needed.
- Acceptance bar for shipping: per-class recall >= 80% on the AK test
  set and >= 70% on the shake-only test set, confusion matrix attached
  to the model folder README. If one class misses, ship three classes
  and say so.

## 5. Packaging and handoff

1. Export with `scripts/export_compact_weights_for_tfjs.py` and
   `scripts/export_standard_weights_for_tfjs.py` (both infer the class
   count from the checkpoint; no change needed).
2. Create `models/compact-v3/` and `models/standard-v2/` with
   `metadata.json` listing the four `class_names` in output order and
   the corrected `instrument_types`. CLUE treats any class named
   `"Noise"` specially, so keep that exact spelling.
3. Add a README per folder with provenance, dataset counts, and the
   confusion matrices.
4. Open a PR, tag Scott Cytacki and Teale Fristoe, and post in the
   `seismic-ml-dev` Slack channel so they can load it on a CLUE branch.

## 6. Timeline (three weeks, one person)

| week | dates          | deliverable                                                   |
|------|----------------|---------------------------------------------------------------|
| 0    | Sep 8-12       | Meeting with Concord; class list and fallback agreed; inventory counts; instrument_types fix PR |
| 1    | Sep 15-19      | `build_dataset.py`; blast and ice-quake datasets downloaded and reviewed; shake EQ/noise added |
| 2    | Sep 22-26      | Fix split and class map in trainer; train compact and standard; evaluate on both test sets |
| 3    | Sep 29-Oct 3   | Export, package, PR, handoff; Concord loads on a branch |
| 4-5  | Oct 6-17       | Fixes from classroom-style testing; final model pinned before Nov 1 |

Fallback decisions and when to make them:

- End of week 0: ice-quake count too low -> switch fourth class to
  Distant earthquake.
- End of week 2: standard model misses the bar on one class -> ship
  three classes, keep the fourth as a stretch for October.
