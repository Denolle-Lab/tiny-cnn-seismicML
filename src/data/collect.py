"""
Shared plumbing for the data-collection scripts (no torch).

Both the continuous-window collector (``scripts/collect_continuous_windows.py``,
Raspberry Shake stations, minute-aligned windows) and the AK earthquake
collector (event-driven, P-aligned windows) need the same pieces:

- an FDSN client with retries that survive the Raspberry Shake server's
  502s and truncated responses
- the preprocessing chain the training data was built with
- the on-disk layout ``<prefix>_{waveforms,labels,metadata}_<stamp>``
  that ``notebooks/03_training/train_cnn_multiclass.ipynb`` reads
- small pure helpers (minute alignment, local calendar days, catalog
  overlap, burst relabeling) that are unit-tested in tests/test_collect.py

Keep this module free of torch and of anything script-specific.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from .labels import LABEL_MAP

SAMPLING_RATE = 100.0
WINDOW_SEC = 60.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLING_RATE)
EDGE_SEC = 2.0            # taper length and keep-out margin at each end of a contiguous segment
BANDPASS = (2.0, 20.0)    # training band, notebooks/02_labeling/download_AK_only_data.ipynb

FDSN_SERVERS = {
    'AM': 'https://data.raspberryshake.org',
    'AK': 'IRIS',
}


# ----------------------------------------------------------------------------
# FDSN access
# ----------------------------------------------------------------------------

def make_client(network=None, fdsn=None, timeout=120):
    """FDSN client for a network (AM -> Raspberry Shake, AK -> IRIS) or an explicit base URL / name."""
    from obspy.clients.fdsn import Client
    target = fdsn or FDSN_SERVERS.get(network, 'IRIS')
    return Client(base_url=target, timeout=timeout) if target.startswith('http') else Client(target, timeout=timeout)


def with_retries(fn, what, attempts=3, wait=15.0):
    """Call fn(); on any exception wait and retry. Re-raises after the last attempt."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as err:
            if attempt == attempts:
                raise
            print(f'  {what}: {type(err).__name__} (attempt {attempt}/{attempts}), retrying')
            time.sleep(wait * attempt)


def pick_vertical_channel(client, network, station, t0, t1, requested=None):
    """(channel code, (lat, lon, sample_rate)) for the requested or first vertical channel."""
    inv = with_retries(lambda: client.get_stations(network=network, station=station, starttime=t0, endtime=t1,
                                                   level='channel'), f'{network}.{station} metadata')
    chans = {ch.code: (sta.latitude, sta.longitude, ch.sample_rate)
             for net in inv for sta in net for ch in sta}
    if requested:
        if requested not in chans:
            raise ValueError(f'{network}.{station} has no {requested}; channels: {sorted(chans)}')
        return requested, chans[requested]
    for code in ('EHZ', 'SHZ', 'HHZ', 'BHZ', 'ENZ'):
        if code in chans:
            return code, chans[code]
    raise ValueError(f'{network}.{station} has no vertical channel; channels: {sorted(chans)}')


def fetch_stream(client, network, station, channel, t0, t1, chunk_hours=6.0, native_sr=100.0,
                 min_fraction=0.95, pad=5.0, location='*'):
    """
    Download [t0, t1] in chunks and return gap-free traces (Stream.split()).

    A chunk with fewer than ``min_fraction`` of the expected samples counts as
    a failed attempt and is re-requested; the longest response is kept. The
    Raspberry Shake server returns truncated bodies under load.
    """
    from obspy import Stream
    from obspy.clients.fdsn.header import FDSNNoDataException
    out = Stream()
    t = t0
    step = chunk_hours * 3600.0
    while t < t1:
        te = min(t + step, t1)
        expected = (te - t) * native_sr
        best = None
        for attempt in range(1, 4):
            try:
                st = client.get_waveforms(network, station, location, channel, t - pad, te + pad)
            except FDSNNoDataException:
                print(f'  {t.isoformat()[:19]} -> {te.isoformat()[:19]}: no data')
                break
            except Exception as err:
                print(f'  {t.isoformat()[:19]} -> {te.isoformat()[:19]}: {type(err).__name__} (attempt {attempt}/3)')
                time.sleep(10 * attempt)
                continue
            n = sum(len(tr) for tr in st)
            if best is None or n > sum(len(tr) for tr in best):
                best = st
            if n >= min_fraction * expected:
                break
            print(f'  {t.isoformat()[:19]} -> {te.isoformat()[:19]}: {n} of ~{expected:.0f} samples '
                  f'(attempt {attempt}/3), re-requesting')
            time.sleep(10 * attempt)
        if best is not None:
            out += best
            print(f'  {t.isoformat()[:19]} -> {te.isoformat()[:19]}: {sum(len(tr) for tr in best)} samples')
        t = te
    if len(out) == 0:
        return out
    out.merge(method=1, fill_value=None)  # gaps become masked ...
    return out.split()                    # ... and then separate gap-free traces


def catalog_events(t0, t1, lat, lon, minmag=2.0, radius_deg=2.0, client_name='USGS'):
    """Catalogued events in [t0, t1] near (lat, lon): [(origin UTCDateTime, mag, dist_km)], nearest first."""
    from obspy.geodetics import gps2dist_azimuth
    try:
        cat = make_client(fdsn=client_name, timeout=60).get_events(
            starttime=t0, endtime=t1, latitude=lat, longitude=lon, maxradius=radius_deg, minmagnitude=minmag)
    except Exception as err:  # no events (204) or catalog down: nothing to exclude
        print(f'  catalog query: {type(err).__name__}, no events')
        return []
    out = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or ev.magnitudes[0]
        out.append((o.time, m.mag, gps2dist_azimuth(lat, lon, o.latitude, o.longitude)[0] / 1000))
    return sorted(out, key=lambda q: q[2])


# ----------------------------------------------------------------------------
# Preprocessing (matches the AK notebook: detrend, demean, bandpass, 100 Hz)
# ----------------------------------------------------------------------------

def preprocess(tr, freqmin=BANDPASS[0], freqmax=BANDPASS[1], edge_sec=EDGE_SEC):
    """
    Training-data preprocessing on one gap-free trace; returns a copy.

    Fixed-length taper: a percentage of a day-long trace would eat whole
    windows, and callers skip windows within ``edge_sec`` of a segment end.
    """
    tr = tr.copy()
    tr.data = tr.data.astype(np.float64)  # miniSEED ints would overflow in x**2
    tr.detrend('linear')
    tr.detrend('demean')
    tr.taper(max_percentage=None, max_length=edge_sec)
    if tr.stats.sampling_rate != SAMPLING_RATE:
        tr.resample(SAMPLING_RATE)
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4)
    return tr


def detrend_resample(tr):
    """Detrend, demean and resample only (for feature computation on unfiltered data); returns a copy."""
    tr = tr.copy()
    tr.data = tr.data.astype(np.float64)
    tr.detrend('linear').detrend('demean')
    if tr.stats.sampling_rate != SAMPLING_RATE:
        tr.resample(SAMPLING_RATE)
    return tr


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------

def window_starts(n_samples, sr, offset_sec, window_samples=WINDOW_SAMPLES, window_sec=WINDOW_SEC,
                  edge_sec=EDGE_SEC):
    """
    Start samples of non-overlapping windows aligned to multiples of ``window_sec``
    counted from a global origin, given a segment of ``n_samples`` that begins
    ``offset_sec`` after that origin. Windows within ``edge_sec`` of either end
    are skipped (taper and filter transient), and a half-sample tolerance
    absorbs the ~2 ms sample-phase offset of Raspberry Shake data.
    """
    rem = offset_sec % window_sec
    tol = 0.5 / sr
    first = 0 if (rem < tol or window_sec - rem < tol) else int(round((window_sec - rem) * sr))
    edge = int(edge_sec * sr)
    starts = range(first, n_samples - window_samples + 1, window_samples)
    return [s for s in starts if s >= edge and s + window_samples <= n_samples - edge]


def local_day_bounds(day, tz):
    """UTC (t0, t1) for the local calendar day ``day`` ('YYYY-MM-DD') in ``tz``; DST-safe (23 or 25 h days)."""
    tz = ZoneInfo(tz) if isinstance(tz, str) else tz
    utc = ZoneInfo('UTC')
    local0 = datetime.fromisoformat(day).replace(tzinfo=tz)
    local1 = (local0 + timedelta(days=1)).replace(tzinfo=None).replace(tzinfo=tz)
    return UTCDateTime(local0.astimezone(utc)), UTCDateTime(local1.astimezone(utc))


def local_hour(t, tz):
    """Local hour of day (float) of a UTCDateTime in ``tz``."""
    tz = ZoneInfo(tz) if isinstance(tz, str) else tz
    lt = t.datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)
    return lt.hour + lt.minute / 60 + lt.second / 3600


def overlaps_event(w_t0, w_t1, quakes, coda_sec=120.0):
    """True if window [w_t0, w_t1] overlaps [origin, origin + coda_sec] of any (origin, mag, dist) in quakes."""
    return any(q_t <= w_t1 and q_t + coda_sec >= w_t0 for q_t, _, _ in quakes)


def relabel_bursts(meta, positive_label, factor=20.0, noise_label=0):
    """
    Night windows whose raw ``peak`` is at least ``factor`` times the station's
    median night ``rms`` are single vehicle passes: relabel them to
    ``positive_label`` and tag ``label_method`` with ``+burst>=<factor>x``.
    Adds ``night_rms_ref`` and ``burst_ratio`` columns. Returns the count per station.
    """
    night = meta['label'] == noise_label
    ref = meta[night].groupby('station')['rms'].median()
    meta['night_rms_ref'] = meta['station'].map(ref)
    meta['burst_ratio'] = meta['peak'] / meta['night_rms_ref']
    burst = night & (meta['burst_ratio'] >= factor)
    meta.loc[burst, 'label'] = positive_label
    meta.loc[burst, 'label_name'] = LABEL_MAP[positive_label]
    meta.loc[burst, 'window_type'] = LABEL_MAP[positive_label].lower()
    meta.loc[burst, 'label_method'] = meta.loc[burst, 'label_method'] + f'+burst>={factor:g}x'
    return burst.groupby(meta['station']).sum().astype(int).to_dict()


# ----------------------------------------------------------------------------
# On-disk layout
# ----------------------------------------------------------------------------

def write_dataset(out_dir, prefix, X, y, meta, summary_lines=(), stamp=None):
    """
    Write ``<prefix>_waveforms_<stamp>.npy`` (fixed (N, L) float32, or object
    array for ragged windows), ``<prefix>_labels_<stamp>.npy`` (global label
    ints), ``<prefix>_metadata_<stamp>.csv`` and ``<prefix>_summary_<stamp>.txt``.
    Returns a dict of the four paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    y = np.asarray(y, dtype=np.int64)
    meta = pd.DataFrame(meta)
    if len(meta) != len(y) or len(X) != len(y):
        raise ValueError(f'lengths differ: waveforms {len(X)}, labels {len(y)}, metadata {len(meta)}')
    paths = {k: out_dir / f'{prefix}_{k}_{stamp}.{ext}'
             for k, ext in (('waveforms', 'npy'), ('labels', 'npy'), ('metadata', 'csv'), ('summary', 'txt'))}
    if isinstance(X, np.ndarray) and X.dtype != object:
        np.save(paths['waveforms'], X.astype(np.float32))
    else:  # ragged windows (AK earthquake windows are longer than noise windows)
        arr = np.empty(len(X), dtype=object)
        for i, w in enumerate(X):
            arr[i] = np.asarray(w, dtype=np.float32)
        np.save(paths['waveforms'], arr, allow_pickle=True)
    np.save(paths['labels'], y)
    meta.to_csv(paths['metadata'], index=False)
    counts = (meta.groupby(['station', 'label_name']).size().unstack(fill_value=0)
              if {'station', 'label_name'} <= set(meta.columns) else None)
    with open(paths['summary'], 'w') as f:
        f.write(f'Dataset {prefix}\n{"=" * 60}\n\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        for line in summary_lines:
            f.write(line.rstrip() + '\n')
        if counts is not None:
            f.write('\nWindows per station and label:\n' + counts.to_string() + '\n')
        f.write(f'\nFiles:\n  {paths["waveforms"].name}  n={len(y)}\n  {paths["labels"].name}\n  {paths["metadata"].name}\n')
    return paths


def latest_dataset(data_dir, prefix):
    """(waveforms, labels, metadata) paths of the newest stamp for ``prefix``; metadata may be None."""
    data_dir = Path(data_dir)
    wfs = sorted(data_dir.glob(f'{prefix}_waveforms_*.npy'))
    if not wfs:
        raise FileNotFoundError(f'No {prefix}_waveforms_*.npy in {data_dir}')
    stamp = wfs[-1].stem.split('_waveforms_')[-1]
    lf = data_dir / f'{prefix}_labels_{stamp}.npy'
    if not lf.exists():
        raise FileNotFoundError(f'No labels file with stamp {stamp} next to {wfs[-1].name}')
    mf = data_dir / f'{prefix}_metadata_{stamp}.csv'
    return wfs[-1], lf, (mf if mf.exists() else None)
