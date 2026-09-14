"""Tests for the pure helpers in src/data/collect.py (numpy, pandas, obspy; no network, no torch)."""

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime

from src.data import collect
from src.data.labels import LABEL_MAP


def test_src_data_imports_without_torch(monkeypatch):
    # src.data must stay importable where torch is absent (collection machines)
    import importlib, sys
    monkeypatch.setitem(sys.modules, 'torch', None)
    for name in [m for m in list(sys.modules) if m == 'src' or m.startswith('src.')]:
        monkeypatch.delitem(sys.modules, name)
    mod = importlib.import_module('src.data')
    assert mod.LABEL_MAP[2] == 'Earthquake'
    with pytest.raises(ImportError):
        mod.SeismicDataset  # noqa: B018  (needs torch)


def test_window_starts_aligns_to_minutes_and_keeps_clear_of_edges():
    sr = 100.0
    n = int(10 * 60 * sr) + 1000           # 10 minutes plus 10 s
    starts = collect.window_starts(n, sr, offset_sec=55.0)     # segment begins 55 s after the origin
    assert starts[0] == 500                 # first whole minute is 5 s in
    assert all((s - 500) % 6000 == 0 for s in starts)
    assert starts[-1] + 6000 <= n - int(collect.EDGE_SEC * sr)
    assert len(starts) == 10


def test_window_starts_half_sample_tolerance_and_edge_keepout():
    sr = 100.0
    n = int(3 * 60 * sr) + 1000
    # Raspberry Shake style: segment starts 2 ms before the whole minute. The
    # aligned window would start at sample 0, which is inside the 2 s keep-out,
    # so the first kept window is the next minute (same as an exact start).
    assert collect.window_starts(n, sr, offset_sec=59.998)[0] == 6000
    assert collect.window_starts(n, sr, offset_sec=0.0)[0] == 6000
    # A chunk fetched with the usual 5 s pad starts 5 s early: first window at sample 500
    assert collect.window_starts(n, sr, offset_sec=55.0)[0] == 500
    assert collect.window_starts(n, sr, offset_sec=54.998)[0] == 500   # 2 ms early, rounds to the same sample


def test_local_day_bounds_handles_dst():
    t0, t1 = collect.local_day_bounds('2026-09-09', 'America/Anchorage')
    assert t0 == UTCDateTime('2026-09-09T08:00:00') and t1 - t0 == 24 * 3600
    t0, t1 = collect.local_day_bounds('2026-11-01', 'America/Anchorage')  # fall back: 25 h local day
    assert t1 - t0 == 25 * 3600


def test_overlaps_event():
    q = [(UTCDateTime('2026-09-09T18:35:05'), 2.9, 40.0)]
    w0 = UTCDateTime('2026-09-09T18:35:00')
    assert collect.overlaps_event(w0, w0 + 60, q, coda_sec=120)          # origin inside window
    assert collect.overlaps_event(w0 + 120, w0 + 180, q, coda_sec=120)   # coda (to 18:37:05) still inside
    assert not collect.overlaps_event(w0 - 120, w0 - 60, q, coda_sec=120)  # window ends before the origin
    assert not collect.overlaps_event(w0 + 180, w0 + 240, q, coda_sec=120)  # window starts after the coda
    assert not collect.overlaps_event(w0, w0 + 60, [], coda_sec=120)


def test_relabel_bursts_per_station():
    meta = pd.DataFrame({
        'station': ['A'] * 4 + ['B'] * 4,
        'label': [0, 0, 0, 1, 0, 0, 0, 1],
        'label_name': ['Noise', 'Noise', 'Noise', 'Traffic'] * 2,
        'window_type': ['noise', 'noise', 'noise', 'traffic'] * 2,
        'label_method': ['timeofday'] * 8,
        'rms': [100, 100, 120, 900, 300, 300, 310, 900],
        'peak': [400, 2500, 500, 5000, 1200, 6500, 1300, 5000],
    })
    counts = collect.relabel_bursts(meta, positive_label=1, factor=20.0)
    assert counts == {'A': 1, 'B': 1}
    assert meta.loc[1, 'label'] == 1 and meta.loc[1, 'label_method'] == 'timeofday+burst>=20x'
    assert meta.loc[5, 'label'] == 1                 # 6500 / 300 = 21.7x at station B
    assert meta.loc[0, 'label'] == 0 and meta.loc[4, 'label'] == 0
    assert meta.loc[3, 'label'] == 1 and meta.loc[3, 'label_method'] == 'timeofday'  # day windows untouched


def test_write_dataset_roundtrip_fixed_and_ragged(tmp_path):
    X = np.random.default_rng(0).standard_normal((5, 6000)).astype(np.float32)
    y = np.array([0, 0, 1, 0, 1])
    meta = pd.DataFrame({'station': ['S'] * 5, 'label': y, 'label_name': [LABEL_MAP[v] for v in y]})
    paths = collect.write_dataset(tmp_path, 'AM_test', X, y, meta, summary_lines=['hello'], stamp='20260101_000000')
    assert np.array_equal(np.load(paths['waveforms']), X)
    assert np.array_equal(np.load(paths['labels']), y)
    assert len(pd.read_csv(paths['metadata'])) == 5
    assert 'hello' in paths['summary'].read_text()
    wf, lf, mf = collect.latest_dataset(tmp_path, 'AM_test')
    assert (wf, lf, mf) == (paths['waveforms'], paths['labels'], paths['metadata'])

    ragged = [np.zeros(6000), np.zeros(12000)]
    p2 = collect.write_dataset(tmp_path, 'AK_test', ragged, [0, 2], pd.DataFrame({'station': ['S', 'S']}), stamp='x')
    arr = np.load(p2['waveforms'], allow_pickle=True)
    assert arr.dtype == object and len(arr[1]) == 12000

    with pytest.raises(ValueError):
        collect.write_dataset(tmp_path, 'bad', X, y[:3], meta)
