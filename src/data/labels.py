"""
Shared label scheme for every labeling notebook and training run.

Two ideas live here:

1. A global integer label written to disk by the labeling notebooks
   (``*_labels_*.npy``). These integers never change: 0 and 2 are already
   on disk in ``notebooks/02_labeling/labeled_data/``, and 1 is the
   rule-based traffic class. New classes append at the end.

2. A per-model class subset. Each trained model sees only the classes it is
   meant to separate, remapped to contiguous output indices (0..K-1) at load
   time so ``CrossEntropyLoss`` and the browser export both work unchanged.
   The subset order is the model's output order and is what gets written to
   ``class_names`` in the checkpoint and in ``models/<id>/metadata.json``.

See GitHub issue #13 for the roadmap this implements.
"""

import numpy as np

# Global label integer -> class name. Append only; never renumber.
LABEL_MAP = {
    0: 'Noise',
    1: 'Traffic',
    2: 'Earthquake',
    3: 'Avalanche',
    4: 'Train',
    5: 'Aircraft',
}

NAME_TO_LABEL = {name.lower(): label for label, name in LABEL_MAP.items()}

# Named class subsets, one per deployable model. Order = model output order.
MODEL_CLASSES = {
    # What ships today (models/compact-v2, models/standard-v1).
    'earthquake': ['Noise', 'Earthquake'],
    # CLUE WaveRunner "Natural" model.
    'natural': ['Noise', 'Earthquake', 'Avalanche'],
    # CLUE WaveRunner "Human" model.
    'human': ['Noise', 'Train', 'Aircraft'],
    # Human model with the rule-based Raspberry Shake traffic class included.
    'human_traffic': ['Noise', 'Traffic', 'Train', 'Aircraft'],
    # Earlier three-class Raspberry Shake experiment.
    'rule_based': ['Noise', 'Traffic', 'Earthquake'],
}


def _to_name(item):
    """Accept a class name (any case) or a global label integer; return the canonical name."""
    if isinstance(item, (int, np.integer)):
        if int(item) not in LABEL_MAP:
            raise KeyError(f"Unknown label integer {item}; known: {LABEL_MAP}")
        return LABEL_MAP[int(item)]
    key = str(item).lower()
    if key not in NAME_TO_LABEL:
        raise KeyError(f"Unknown class name '{item}'; known: {list(LABEL_MAP.values())}")
    return LABEL_MAP[NAME_TO_LABEL[key]]


def class_subset(classes):
    """
    Resolve a class specification to an ordered list of canonical class names.

    Args:
        classes: one of
            - a key of ``MODEL_CLASSES`` (e.g. ``'natural'``),
            - a list of class names (any case) or global label integers,
            - ``None``, meaning "every class in ``LABEL_MAP``".

    Returns:
        list[str]: class names in model output order.
    """
    if classes is None:
        return [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
    if isinstance(classes, str):
        if classes not in MODEL_CLASSES:
            raise KeyError(f"Unknown model class set '{classes}'; known: {list(MODEL_CLASSES)}")
        return list(MODEL_CLASSES[classes])
    names = [_to_name(c) for c in classes]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate classes in {classes}")
    return names


def classes_from_config(model_cfg):
    """
    Class specification from a config ``model`` block.

    ``classes`` wins. A legacy ``num_classes`` alone maps 2 -> ``'earthquake'``
    (the deployed AK model) and 3 -> ``'rule_based'``; anything else is the
    first N classes of ``LABEL_MAP``.
    """
    if model_cfg.get('classes') is not None:
        return model_cfg['classes']
    n = int(model_cfg.get('num_classes', 2))
    if n < 1 or n > len(LABEL_MAP):
        raise ValueError(
            f"num_classes={n} does not match a known subset; set model.classes to a "
            f"MODEL_CLASSES key ({', '.join(MODEL_CLASSES)}) or a list of class names."
        )
    return {2: 'earthquake', 3: 'rule_based'}.get(n, list(range(n)))


def global_labels(classes):
    """Global label integers for a class specification, in model output order."""
    return [NAME_TO_LABEL[name.lower()] for name in class_subset(classes)]


def infer_classes(labels):
    """
    Class names present in an on-disk label array, ordered by global label integer.

    This is the back-compatible path for datasets written before per-model
    subsets existed: a file holding only {0, 2} yields ``['Noise', 'Earthquake']``.
    """
    present = sorted(int(v) for v in np.unique(labels))
    return [_to_name(v) for v in present]


def select_classes(waveforms, labels, classes, allow_missing=False):
    """
    Keep only the windows whose global label is in ``classes`` and remap the
    labels to contiguous model output indices.

    Args:
        waveforms (np.ndarray or sequence): windows, first axis = sample.
        labels (np.ndarray): global label integers, shape (N,).
        classes: class specification accepted by ``class_subset``.
        allow_missing (bool): if False (default), raise when one of the
            requested classes has no samples, since a zero-count class makes
            class-weighted loss and the confusion matrix meaningless.

    Returns:
        tuple: (waveforms_subset, labels_remapped, class_names)
            ``labels_remapped[i]`` is the index of that window's class in
            ``class_names``.
    """
    labels = np.asarray(labels)
    names = class_subset(classes)
    wanted = global_labels(names)

    mask = np.isin(labels, wanted)
    lut = np.full(max(LABEL_MAP) + 1, -1, dtype=np.int64)  # global label -> output index
    lut[wanted] = np.arange(len(wanted))
    y_new = lut[labels[mask].astype(np.int64)]

    counts = np.bincount(y_new, minlength=len(names))
    missing = [n for n, c in zip(names, counts) if c == 0]
    if missing and not allow_missing:
        raise ValueError(
            f"No samples for {missing} in this dataset. Labels present: "
            f"{infer_classes(labels)}. Pass allow_missing=True to train anyway."
        )

    if isinstance(waveforms, np.ndarray):
        x_new = waveforms[mask]
    else:  # ragged list of windows (AK notebook keeps variable-length arrays)
        x_new = [w for w, keep in zip(waveforms, mask) if keep]

    return x_new, y_new, names
