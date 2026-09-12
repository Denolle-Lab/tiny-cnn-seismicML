"""Tests for the shared label scheme in src/data/labels.py (numpy only)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the module directly so the tests run without torch/scipy installed.
_spec = importlib.util.spec_from_file_location(
    "labels", Path(__file__).resolve().parents[1] / "src" / "data" / "labels.py"
)
labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels)


def test_on_disk_integers_are_stable():
    # These three are already written to labeled_data/ by the labeling notebooks.
    assert labels.LABEL_MAP[0] == 'Noise'
    assert labels.LABEL_MAP[1] == 'Traffic'
    assert labels.LABEL_MAP[2] == 'Earthquake'


def test_every_model_subset_resolves():
    for key, names in labels.MODEL_CLASSES.items():
        assert labels.class_subset(key) == names
        assert names[0] == 'Noise', f"{key}: noise must be output index 0"


def test_class_subset_accepts_names_and_ints():
    assert labels.class_subset(['noise', 'EARTHQUAKE']) == ['Noise', 'Earthquake']
    assert labels.class_subset([0, 2]) == ['Noise', 'Earthquake']
    assert labels.global_labels('natural') == [0, 2, 3]
    with pytest.raises(KeyError):
        labels.class_subset('not-a-model')
    with pytest.raises(KeyError):
        labels.class_subset(['Noise', 'Volcano'])
    with pytest.raises(ValueError):
        labels.class_subset(['Noise', 'noise'])


def test_infer_classes_matches_old_notebook_behaviour():
    # AK files hold {0, 2}; the old notebook remapped that to Noise/Earthquake.
    assert labels.infer_classes(np.array([2, 0, 2, 0])) == ['Noise', 'Earthquake']
    assert labels.infer_classes(np.array([0, 1, 2])) == ['Noise', 'Traffic', 'Earthquake']


def test_select_classes_remaps_contiguously_and_drops_others():
    y = np.array([0, 2, 0, 2, 3, 1, 4])
    X = np.arange(len(y) * 2).reshape(len(y), 2)
    Xs, ys, names = labels.select_classes(X, y, 'natural')
    assert names == ['Noise', 'Earthquake', 'Avalanche']
    assert ys.tolist() == [0, 1, 0, 1, 2]
    assert Xs.tolist() == X[[0, 1, 2, 3, 4]].tolist()
    assert ys.dtype == np.int64


def test_select_classes_raises_on_empty_class():
    y = np.array([0, 2, 0])
    X = np.zeros((3, 4))
    with pytest.raises(ValueError, match="Avalanche"):
        labels.select_classes(X, y, 'natural')
    _, ys, _ = labels.select_classes(X, y, 'natural', allow_missing=True)
    assert ys.tolist() == [0, 1, 0]


def test_select_classes_handles_ragged_lists():
    y = np.array([0, 2, 1])
    X = [np.zeros(10), np.zeros(12), np.zeros(11)]
    Xs, ys, names = labels.select_classes(X, y, 'earthquake')
    assert isinstance(Xs, list) and len(Xs) == 2
    assert ys.tolist() == [0, 1]


def test_classes_from_config_prefers_classes_and_maps_legacy_num_classes():
    assert labels.classes_from_config({'classes': 'natural'}) == 'natural'
    assert labels.classes_from_config({'classes': ['Noise', 'Train']}) == ['Noise', 'Train']
    assert labels.class_subset(labels.classes_from_config({'num_classes': 2})) == ['Noise', 'Earthquake']
    assert labels.class_subset(labels.classes_from_config({'num_classes': 3})) == ['Noise', 'Traffic', 'Earthquake']
    assert labels.class_subset(labels.classes_from_config({})) == ['Noise', 'Earthquake']


def test_select_classes_vectorized_remap_matches_lookup():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2, 3, 4, 5], size=10_000)
    X = np.zeros((len(y), 1))
    _, ys, names = labels.select_classes(X, y, 'human_traffic')
    expect = {0: 0, 1: 1, 4: 2, 5: 3}
    assert ys.tolist() == [expect[int(v)] for v in y if int(v) in expect]
    assert names == ['Noise', 'Traffic', 'Train', 'Aircraft']


def test_classes_from_config_rejects_unknown_num_classes():
    with pytest.raises(ValueError, match="model.classes"):
        labels.classes_from_config({'num_classes': 9})
    assert labels.class_subset(labels.classes_from_config({'num_classes': 6})) == list(labels.LABEL_MAP.values())
