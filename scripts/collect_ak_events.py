#!/usr/bin/env python3
"""
Pull AK (Alaska Seismic Network) earthquake and pre-event noise windows into
the labeled-data layout (``AK_waveforms_<stamp>.npy``, ``AK_labels_<stamp>.npy``,
``AK_metadata_<stamp>.csv``, ``AK_summary_<stamp>.txt``) so they load in
``notebooks/03_training/train_cnn_multiclass.ipynb`` next to the AM sets.

Script form of notebooks/02_labeling/download_AK_only_data.ipynb. Events come
from the USGS catalog, P arrivals from the ComCat phase-data product (reviewed
picks), waveforms from IRIS. Windows follow the random-shift scheme the
training notebook expects: earthquake = [P - eq_pre_sec, P + eq_post_sec]
(120 s, P at 30 s, ``p_offset_sec``), noise = [P - noise_pre_sec, P - noise_end_sec]
(60 s). Preprocessing, FDSN retries and file writing are shared with
scripts/collect_continuous_windows.py via src/data/collect.py.

Waveforms are saved as bandpassed counts (float32), like the AM sets; the
training notebook z-scores every crop. ``--zscore`` (or ``zscore: true`` in the
config) normalizes each window on disk instead, which together with
``--drop-manual`` reproduces the July 2026 ``AK_*`` files byte for byte.

Settings resolve in three layers: built-in defaults, then the YAML given with
``--config``, then any flag typed on the command line. So
``--config configs/ak_events.yaml --max-events 5`` is a quick test of the
configured pull. (The AM collector lets the config win over flags; this
script lets typed flags win so test runs need no YAML edits.)

Example (from repo root):
  python scripts/collect_ak_events.py --config configs/ak_events.yaml
"""

import argparse
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from libcomcat.search import get_event_by_id  # reviewed P picks from the ComCat phase-data product
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import classic_sta_lta
from scipy.stats import kurtosis
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# src.data imports without torch (the PyTorch Dataset is loaded lazily).
from src.data.labels import LABEL_MAP, NAME_TO_LABEL
from src.data.collect import SAMPLING_RATE, detrend_resample, make_client, preprocess, with_retries, write_dataset

EARTHQUAKE, NOISE = NAME_TO_LABEL['earthquake'], NAME_TO_LABEL['noise']
REGION_KEYS = ('min_lat', 'max_lat', 'min_lon', 'max_lon')

# Built-in defaults = the notebook's configuration cell. configs/ak_events.yaml
# repeats them so the YAML is self-describing.
DEFAULTS = {
    'network': 'AK',
    'prefix': 'AK',
    'start': '2020-01-01',
    'end': '2025-12-01',
    'min_magnitude': 3.0,
    'max_magnitude': 7.0,
    'max_events': 700,
    'max_station_distance_km': 150.0,
    'max_stations_per_event': 10,
    'channels': ['BHZ', 'HHZ'],
    'station_pad_deg': 0.5,
    'freqmin': 2.0,
    'freqmax': 20.0,
    'eq_pre_sec': 30.0,
    'eq_post_sec': 90.0,
    'noise_pre_sec': 90.0,
    'noise_end_sec': 30.0,
    'zscore': False,          # filtered counts on disk like the AM sets; True = per-window z-score, the July 2026 format
    'drop_manual': False,     # drop the windows rejected by eye in the July 2026 review
    'drop_list': 'docs/ak_dropped_windows.csv',  # (event_id, station, window_type) rows; relative to the repo root
    'request_delay_sec': 0.2,
    'fdsn': None,
    'out': str(REPO_ROOT / 'notebooks' / '02_labeling' / 'labeled_data'),
    'region': dict(zip(REGION_KEYS, (61.0157, 61.2795, -150.2953, -149.6063))),
}


def parse_args():
    """Flags default to SUPPRESS so only flags actually typed appear in the namespace."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                argument_default=argparse.SUPPRESS)
    p.add_argument('--config', help='YAML with region: and collection: blocks (configs/ak_events.yaml)')
    p.add_argument('--network')
    p.add_argument('--prefix', help='Output file prefix (default AK)')
    p.add_argument('--start', help='Catalog search start, UTC')
    p.add_argument('--end', help='Catalog search end, UTC')
    p.add_argument('--min-magnitude', type=float)
    p.add_argument('--max-magnitude', type=float)
    p.add_argument('--max-events', type=int, help='Events processed, newest first')
    p.add_argument('--max-station-distance-km', type=float)
    p.add_argument('--max-stations-per-event', type=int)
    p.add_argument('--channels', nargs='+', help='Vertical channels in order of preference')
    p.add_argument('--station-pad-deg', type=float, help='Station search box = region box +/- this')
    p.add_argument('--region', nargs=4, type=float, metavar=tuple(k.upper() for k in REGION_KEYS))
    p.add_argument('--freqmin', type=float)
    p.add_argument('--freqmax', type=float)
    p.add_argument('--eq-pre-sec', type=float)
    p.add_argument('--eq-post-sec', type=float)
    p.add_argument('--noise-pre-sec', type=float)
    p.add_argument('--noise-end-sec', type=float)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--zscore', dest='zscore', action='store_true',
                   help='Z-score each window before saving (the July 2026 AK format; reproduces those files)')
    g.add_argument('--no-zscore', dest='zscore', action='store_false',
                   help='Save filtered counts like the AM collector (default; training z-scores each crop anyway)')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--drop-manual', dest='drop_manual', action='store_true',
                   help='Drop the windows listed in --drop-list (the July 2026 eye review)')
    g.add_argument('--no-drop-manual', dest='drop_manual', action='store_false', help='Keep every window')
    p.add_argument('--drop-list', help='CSV with event_id, station, window_type columns (default docs/ak_dropped_windows.csv)')
    p.add_argument('--request-delay-sec', type=float)
    p.add_argument('--fdsn', help='Waveform FDSN base URL or name; default by network')
    p.add_argument('--out', help='Output directory')
    return p.parse_args()


def resolve_settings(args):
    """defaults <- config file <- typed flags. Returns a plain dict."""
    cfg = dict(DEFAULTS)
    config_path = getattr(args, 'config', None)
    if config_path:
        with open(config_path, encoding='utf-8') as fh:
            y = yaml.safe_load(fh) or {}
        col = y.get('collection', {})
        unknown = sorted(set(col) - set(DEFAULTS))
        if unknown:
            raise SystemExit(f'{config_path}: unknown collection keys {unknown}')
        cfg.update(col)
        if 'region' in y:
            cfg['region'] = {k: y['region'][k] for k in REGION_KEYS}
            if 'center_lat' in y['region']:  # notebook used a fixed value for the longitude pad
                cfg['region']['center_lat'] = y['region']['center_lat']
    typed = {k: v for k, v in vars(args).items() if k != 'config'}
    if 'region' in typed:
        typed['region'] = dict(zip(REGION_KEYS, typed['region']))
    cfg.update(typed)
    if not Path(cfg['drop_list']).is_absolute():
        cfg['drop_list'] = str(REPO_ROOT / cfg['drop_list'])
    cfg['config'] = config_path
    return cfg


def drop_manual_windows(windows, rows, drop_list):
    """
    Remove windows whose (event_id, station, window_type) is in the drop list
    and mark the survivors as reviewed. Returns (windows, rows, n_dropped).
    """
    drop = pd.read_csv(drop_list, comment='#')
    keys = set(map(tuple, drop[['event_id', 'station', 'window_type']].astype(str).values))
    tag = f'manual:{Path(drop_list).name}'
    kept_w, kept_r = [], []
    for w, r in zip(windows, rows):
        if (r['event_id'], r['station'], r['window_type']) in keys:
            continue
        r = dict(r, window_id=len(kept_r), reviewed=tag)
        kept_w.append(w)
        kept_r.append(r)
    return kept_w, kept_r, len(windows) - len(kept_w)


def event_id_of(event):
    """USGS event code (e.g. ak2025xmmutb) from an obspy Event resource id, as the notebook parsed it."""
    rid = str(event.resource_id)
    if 'eventid=' in rid:
        return rid.split('eventid=')[1].split('&')[0]
    return rid.split('/')[-1].replace('.quakeml', '')


def find_events(cfg, usgs):
    """
    USGS catalog for the region box padded by max_station_distance_km, newest
    first, cut to max_events. Returns a DataFrame with one row per event.
    """
    r = cfg['region']
    center_lat = r.get('center_lat', 0.5 * (r['min_lat'] + r['max_lat']))
    lat_pad = cfg['max_station_distance_km'] / 111.0
    lon_pad = cfg['max_station_distance_km'] / (111.0 * np.cos(np.radians(center_lat)))
    catalog = with_retries(lambda: usgs.get_events(
        starttime=UTCDateTime(cfg['start']), endtime=UTCDateTime(cfg['end']),
        minlatitude=r['min_lat'] - lat_pad, maxlatitude=r['max_lat'] + lat_pad,
        minlongitude=r['min_lon'] - lon_pad, maxlongitude=r['max_lon'] + lon_pad,
        minmagnitude=cfg['min_magnitude'], maxmagnitude=cfg['max_magnitude'],
        orderby='time'), 'USGS catalog')
    rows = []
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        rows.append({
            'event_id': event_id_of(event),
            'time': origin.time,
            'latitude': origin.latitude,
            'longitude': origin.longitude,
            'depth_km': origin.depth / 1000 if origin.depth else 10.0,  # notebook default when depth is missing
            'magnitude': magnitude.mag,
            'mag_type': magnitude.magnitude_type,
        })
    events = pd.DataFrame(rows)
    print(f'USGS: {len(events)} events M{cfg["min_magnitude"]}-{cfg["max_magnitude"]} '
          f'{cfg["start"]} -> {cfg["end"]} in the padded box; processing the newest {min(cfg["max_events"], len(events))}')
    return events.head(cfg['max_events'])


def find_stations(cfg, iris):
    """
    Stations of cfg.network in the region box padded by station_pad_deg that
    have one of cfg.channels, first channel in preference order wins, first
    epoch of a station wins. Returns a DataFrame with one row per station.
    """
    r = cfg['region']
    pad = cfg['station_pad_deg']
    inv = with_retries(lambda: iris.get_stations(
        network=cfg['network'],
        minlatitude=r['min_lat'] - pad, maxlatitude=r['max_lat'] + pad,
        minlongitude=r['min_lon'] - pad, maxlongitude=r['max_lon'] + pad,
        starttime=UTCDateTime(cfg['start']), endtime=UTCDateTime(cfg['end']),
        channel=','.join(c[:2] + '?' for c in cfg['channels']), level='channel'), f'{cfg["network"]} stations')
    rows, seen = [], set()
    for net in inv:
        for sta in net:
            if sta.code in seen:
                continue
            by_code = {ch.code: ch for ch in sta.channels}
            channel = next((c for c in cfg['channels'] if c in by_code), None)
            if channel is None:
                continue
            rows.append({
                'network': net.code,
                'station': sta.code,
                'latitude': sta.latitude,
                'longitude': sta.longitude,
                'elevation': sta.elevation,
                'channel': channel,
                'native_sampling_rate': by_code[channel].sample_rate,
            })
            seen.add(sta.code)
    stations = pd.DataFrame(rows)
    print(f'{cfg["network"]}: {len(stations)} stations with {"/".join(cfg["channels"])} in the box: '
          + ', '.join(f'{s.station}.{s.channel}@{s.native_sampling_rate:g}Hz' for s in stations.itertuples()))
    return stations


_phase_cache = {}  # event_id -> {STATION: P pick UTCDateTime}; one ComCat download per event


def p_picks(event_id):
    """
    Station code -> P pick time for one event from the ComCat phase-data
    QuakeML, cached per event. Notebook logic: phase labels on the origin's
    arrivals first (older AEC events), pick phase_hint as the fallback.
    """
    if event_id in _phase_cache:
        return _phase_cache[event_id]
    picks = {}
    try:
        products = with_retries(lambda: get_event_by_id(event_id).getProducts('phase-data'),
                                f'{event_id} phase-data')
        if products:
            quakeml_bytes, _ = products[0].getContentBytes('quakeml.xml')
            cat = read_events(BytesIO(quakeml_bytes))
            if cat:
                ev = cat[0]
                origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
                by_id = {str(p.resource_id): p for p in ev.picks}
                if origin is not None:
                    for arr in origin.arrivals:
                        if not (arr.phase or '').upper().startswith('P'):
                            continue
                        pick = by_id.get(str(arr.pick_id))
                        if pick is not None:
                            picks.setdefault(pick.waveform_id.station_code.upper(), pick.time)
                for pick in ev.picks:
                    if (pick.phase_hint or '').upper().startswith('P'):
                        picks.setdefault(pick.waveform_id.station_code.upper(), pick.time)
    except Exception as err:
        print(f'      [comcat ERROR] {event_id}: {type(err).__name__}: {str(err)[:120]}')
    _phase_cache[event_id] = picks
    return picks


def p_arrival_sec(event_id, station, origin_time):
    """Seconds from the catalog origin to the station's P pick, or None; the notebook keeps 0 < p < 300."""
    pick = p_picks(event_id).get(station.upper())
    if pick is None:
        return None
    p = float(pick - origin_time)
    return p if 0 < p < 300 else None


def main():
    cfg = resolve_settings(parse_args())
    print('Resolved settings:')
    for k, v in cfg.items():
        print(f'  {k}: {v}')

    usgs = make_client(fdsn='USGS')
    iris = make_client(network=cfg['network'], fdsn=cfg['fdsn'])
    print(f'\nCatalog: USGS   Waveforms: {iris.base_url}')

    events = find_events(cfg, usgs)
    stations = find_stations(cfg, iris)
    if events.empty or stations.empty:
        raise SystemExit('No events or no stations; check region, magnitude and date range.')

    windows, rows, stats = collect_windows(cfg, events, stations, iris)
    if not windows:
        raise SystemExit('No windows; check picks, station distance and the time range.')

    review_line = 'Manual review: none applied (--drop-manual to apply the July 2026 eye review)'
    if cfg['drop_manual']:
        windows, rows, n_dropped = drop_manual_windows(windows, rows, cfg['drop_list'])
        review_line = f'Manual review: dropped {n_dropped} windows listed in {Path(cfg["drop_list"]).name}'
        print(f'\n{review_line}')
        if not windows:
            raise SystemExit('Every window was in the drop list.')

    meta = pd.DataFrame(rows)
    y = meta['label'].to_numpy(dtype=np.int64)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # shared by the data files and the config copy
    # write_dataset packs the ragged list (12000-sample earthquake, 6000-sample noise) into an object array
    paths = write_dataset(cfg['out'], cfg['prefix'], windows, y, meta, stamp=stamp, summary_lines=[
                     f'Command: {" ".join(sys.argv)}',
                     review_line,
                     f'Catalog: USGS M{cfg["min_magnitude"]}-{cfg["max_magnitude"]} {cfg["start"]} -> {cfg["end"]}, '
                     f'{stats["events_attempted"]} events attempted, {meta["event_id"].nunique()} with windows',
                     f'Windows: earthquake [P-{cfg["eq_pre_sec"]:g}s, P+{cfg["eq_post_sec"]:g}s], '
                     f'noise [P-{cfg["noise_pre_sec"]:g}s, P-{cfg["noise_end_sec"]:g}s] @ {SAMPLING_RATE:g} Hz, '
                     f'bandpass {cfg["freqmin"]}-{cfg["freqmax"]} Hz, z-scored: {cfg["zscore"]}',
                     f'P arrivals: ComCat phase-data picks; skipped {stats["no_pick"]} station-events without a pick, '
                     f'{stats["no_data"]} without waveforms, {stats["errors"]} with download/processing errors',
                 ])
    counts = meta.groupby(['station', 'label_name']).size().unstack(fill_value=0)
    print(f'\nSaved {len(y)} windows -> {paths["waveforms"].name}\n{counts.to_string()}')
    # The settings this run actually used, next to the data (the YAML may change later).
    config_copy = Path(cfg['out']) / f'{cfg["prefix"]}_config_{stamp}.yaml'
    with open(config_copy, 'w', encoding='utf-8') as fh:
        fh.write(f'# Resolved settings of: {" ".join(sys.argv)}\n')
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f'Settings -> {config_copy.name}')


def event_stream(iris, network, station, channel, t0, t1):
    """One waveform request with retries; None when the server has no data (FDSN 204)."""
    def call():
        try:
            return iris.get_waveforms(network, station, '*', channel, t0, t1)
        except FDSNNoDataException:
            return None
    return with_retries(call, f'{network}.{station}.{channel} waveforms')


def raw_features(w_raw, w_band, cft_window):
    """The continuous-window collector's per-window diagnostics, on unnormalized counts."""
    total = float(np.sum(w_raw ** 2)) or 1.0
    return {
        'rms': float(np.sqrt(np.mean(w_raw ** 2))),
        'peak': float(np.max(np.abs(w_raw))),
        'rms_band': float(np.sqrt(np.mean(w_band ** 2))),
        'peak_band': float(np.max(np.abs(w_band))),
        'band_ratio': float(np.sum(w_band ** 2) / total),
        'stalta_band_max': float(np.max(cft_window)),
        'kurtosis_band': float(kurtosis(w_band)),
    }


def station_traces(cfg, iris, st, p_time, rule_band):
    """
    Download [P - noise_pre - 10, P + eq_post + 10] for one station-event and
    return (training trace, raw counts trace, rule-band trace, STA/LTA) or
    None when the server has no data. The training trace follows the notebook
    (merge with interpolation, 5 percent taper, filter, then resample); the raw
    and band traces feed the diagnostic columns the AM collector also writes.
    """
    sr = SAMPLING_RATE
    stream = event_stream(iris, st.network, st.station, st.channel,
                          p_time - cfg['noise_pre_sec'] - 10, p_time + cfg['eq_post_sec'] + 10)
    if stream is None or len(stream) == 0:
        return None
    stream.merge(method=1, fill_value='interpolate')
    tr = preprocess(stream[0], cfg['freqmin'], cfg['freqmax'], taper_pct=0.05, resample_first=False)
    tr_raw = detrend_resample(stream[0])
    tr_band = tr_raw.copy().filter('bandpass', freqmin=rule_band[0], freqmax=rule_band[1], corners=4)
    cft = classic_sta_lta(tr_band.data, int(1.0 * sr), int(10.0 * sr))
    return tr, tr_raw, tr_band, cft


def collect_windows(cfg, events, stations, iris, rule_band=(5.0, 30.0)):
    """
    Notebook main loop: for each event, the nearest stations with a ComCat P
    pick; download [P - noise_pre - 10, P + eq_post + 10], preprocess the
    notebook's way, cut one earthquake and one noise window per station.
    Returns (windows, metadata rows, counters).
    """
    sr = SAMPLING_RATE
    eq_pre, eq_post = int(cfg['eq_pre_sec'] * sr), int(cfg['eq_post_sec'] * sr)
    noise_pre, noise_end = int(cfg['noise_pre_sec'] * sr), int(cfg['noise_end_sec'] * sr)
    eq_len, noise_len = eq_pre + eq_post, noise_pre - noise_end
    bandpass = f'{cfg["freqmin"]}-{cfg["freqmax"]}'
    windows, rows = [], []
    stats = {'events_attempted': len(events), 'no_pick': 0, 'no_data': 0, 'errors': 0, 'station_events': 0}

    for ev in tqdm(list(events.itertuples()), desc='Events', unit='event'):
        dist_km = [gps2dist_azimuth(ev.latitude, ev.longitude, s.latitude, s.longitude)[0] / 1000
                   for s in stations.itertuples()]
        nearby = stations.assign(distance_km=dist_km)
        nearby = nearby[nearby['distance_km'] <= cfg['max_station_distance_km']].sort_values('distance_km')
        line = f'M{ev.magnitude:.1f} {str(ev.time)[:19]} {ev.event_id}'
        if nearby.empty:
            tqdm.write(f'{line}: no stations within {cfg["max_station_distance_km"]:g} km')
            continue

        got = []
        for st in nearby.head(cfg['max_stations_per_event']).itertuples():
            p_sec = p_arrival_sec(ev.event_id, st.station, ev.time)
            if p_sec is None:
                stats['no_pick'] += 1
                continue
            p_time = ev.time + p_sec
            try:
                prepared = station_traces(cfg, iris, st, p_time, rule_band)
            except Exception as err:  # as the notebook: one bad station-event is skipped, not fatal
                stats['errors'] += 1
                tqdm.write(f'    x {st.station}: {type(err).__name__}: {str(err)[:100]}')
                continue
            finally:
                time.sleep(cfg['request_delay_sec'])
            if prepared is None:
                stats['no_data'] += 1
                continue
            tr, tr_raw, tr_band, cft = prepared

            data, trace_start, n = tr.data, tr.stats.starttime, len(tr.data)
            p_sample = int((ev.time + p_sec - trace_start) * sr)
            n_before = len(windows)
            for kind, s, e, length, label, p_offset in (
                    ('earthquake', p_sample - eq_pre, p_sample + eq_post, eq_len, EARTHQUAKE, cfg['eq_pre_sec']),
                    ('noise', p_sample - noise_pre, p_sample - noise_end, noise_len, NOISE, None)):
                if s < 0 or e > n:
                    continue
                w = data[s:e]
                if len(w) != length or np.std(w) <= 1e-10:
                    continue
                if cfg['zscore']:
                    w = (w - np.mean(w)) / (np.std(w) + 1e-10)
                windows.append(np.asarray(w, dtype=np.float32))
                rows.append({
                    # notebook columns, in the notebook's order
                    'event_id': ev.event_id,
                    'event_time': str(ev.time),
                    'magnitude': ev.magnitude,
                    'event_lat': ev.latitude,
                    'event_lon': ev.longitude,
                    'event_depth_km': ev.depth_km,
                    'network': st.network,
                    'station': st.station,
                    'channel': st.channel,
                    'distance_km': st.distance_km,
                    'p_arrival_sec': p_sec,
                    'p_arrival_source': 'comcat',
                    'window_start_sec': s / sr,
                    'window_len_sec': length / sr,
                    'p_offset_sec': p_offset,
                    'window_type': kind,
                    'label': label,
                    'label_name': LABEL_MAP[label],
                    # continuous-window collector columns
                    'window_id': len(rows),
                    'location': tr.stats.location,
                    'latitude': st.latitude,
                    'longitude': st.longitude,
                    'start_time': str(trace_start + s / sr),
                    'end_time': str(trace_start + e / sr),
                    'sampling_rate': sr,
                    'native_sampling_rate': float(st.native_sampling_rate),
                    'label_method': 'comcat:p_pick' if kind == 'earthquake' else 'comcat:pre_p_noise',
                    'source': st.network,
                    'bandpass_hz': bandpass,
                    **raw_features(np.asarray(tr_raw.data[s:e], dtype=np.float64),
                                   np.asarray(tr_band.data[s:e], dtype=np.float64), cft[s:e]),
                    'normalized': bool(cfg['zscore']),
                    'reviewed': '',
                    'review_label': '',
                })
            got.append(f'{st.station} {len(windows) - n_before}')
            stats['station_events'] += 1
        tqdm.write(f'{line}: ' + (', '.join(got) if got else 'no windows'))
    return windows, rows, stats


if __name__ == '__main__':
    main()
