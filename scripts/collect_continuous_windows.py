#!/usr/bin/env python3
"""
Cut continuous seismic data into labeled 60 s / 100 Hz windows in the AK
dataset format (``<prefix>_waveforms_<stamp>.npy``, ``<prefix>_labels_<stamp>.npy``,
``<prefix>_metadata_<stamp>.csv``) so they load in
``notebooks/03_training/train_cnn_multiclass.ipynb`` next to the AK data.

This is the shared machinery for the anthropogenic classes (issues #10, #11,
#12): pull a station for a time range, window it, attach a *provisional*
label, and write a review sheet so a person can confirm the labels by eye
before training.

Label sources (``--label-from``):
  rule       station-relative energy rule: a window is positive when its rms
             is at least ``--rule-rms-factor`` times a reference rms (the
             record's 10th percentile, or ``--reference-rms`` from a quiet
             run of the same station). On AM.R4017 the daytime median rms
             is 4.7x the 02-04 AKDT median while 5-30 Hz band ratio and
             STA/LTA barely move, so sustained energy is the usable cue.
  timeofday  local hour in ``--day-hours`` -> ``--class-name``, in
             ``--night-hours`` -> Noise, anything else dropped. Weak labels
             for a city Raspberry Shake where daytime = traffic (#12).
             Night windows whose raw peak exceeds ``--burst-factor`` times the
             station's median night rms are single vehicle passes (3-6 s,
             5-25 Hz, 30-50x the night envelope at AM.R1796) and get the
             class label too, with label_method ``timeofday+burst``.

``--exclude-events`` drops windows that overlap a catalogued earthquake
(USGS, M >= ``--event-minmag`` within ``--event-radius-deg``, event time to
+``--event-coda-sec``), so daytime windows do not carry unlabeled P and S
arrivals (an M2.9 at 40 km showed up in a "Traffic" window on 2026-09-09).
  events     a CSV of event times (``--events``); windows overlapping an
             event get ``--class-name``, others Noise. Use for train
             timetables (#10) and ADS-B landings/takeoffs (#11)
  all        every window gets ``--class-name`` (e.g. a stretch known to be
             quiet, labeled Noise, or a record you will hand-label)

Preprocessing matches notebooks/02_labeling/download_AK_only_data.ipynb:
merge, linear detrend, demean, bandpass 2-20 Hz (4 corners), resample to
100 Hz. Windows containing a data gap are dropped.

Examples (from repo root):
  # everything listed in the station config (the reproducible pull)
  python scripts/collect_continuous_windows.py --config configs/am_stations.yaml

  python scripts/collect_continuous_windows.py --network AM --station R4017 \
      --start 2026-09-09 --end 2026-09-10 --class-name Traffic \
      --label-from timeofday --review-sheet 24

  python scripts/collect_continuous_windows.py --network AM --station R4017 \
      --start 2026-09-09 --end 2026-09-10 --class-name Train \
      --label-from events --events data/alaska_railroad_passages.csv
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta
from scipy.stats import kurtosis

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Only the label module: src/__init__ pulls in torch, which this script does not need.
import importlib.util
_spec = importlib.util.spec_from_file_location('labels', REPO_ROOT / 'src' / 'data' / 'labels.py')
labels_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels_mod)
LABEL_MAP, NAME_TO_LABEL = labels_mod.LABEL_MAP, labels_mod.NAME_TO_LABEL

SAMPLING_RATE = 100.0
WINDOW_SEC = 60.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLING_RATE)
EDGE_SEC = 2.0  # taper length and keep-out margin at each end of a contiguous segment

FDSN_SERVERS = {
    'AM': 'https://data.raspberryshake.org',
    'AK': 'IRIS',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None,
                   help='Station config YAML (configs/am_stations.yaml): runs every listed day for the '
                        'stations whose role is in collection.roles. Every key under collection: overrides '
                        'the flag of the same name; flags absent from the config apply as given')
    p.add_argument('--network', default=None, help='FDSN network code (AM = Raspberry Shake, AK = Alaska)')
    p.add_argument('--station', default=None, nargs='+', help='One or more station codes')
    p.add_argument('--channel', default=None,
                   help='Channel code (default: first vertical channel found, EHZ/SHZ/HHZ/BHZ)')
    p.add_argument('--start', default=None, help='UTC start, e.g. 2026-09-09T16:00:00 '
                                                 '(config days are local calendar days instead)')
    p.add_argument('--end', default=None, help='UTC end')
    p.add_argument('--chunk-hours', type=float, default=6.0, help='Download chunk length (hours)')
    p.add_argument('--fdsn', default=None, help='FDSN base URL or name; default by network')

    p.add_argument('--class-name', default='Noise', choices=list(LABEL_MAP.values()),
                   help='Class assigned to positive windows')
    p.add_argument('--label-from', default='rule', choices=['rule', 'timeofday', 'events', 'all'])
    p.add_argument('--events', default=None,
                   help='CSV with a "time" column (UTC) and optional "duration_sec", "event_id"')
    p.add_argument('--event-pad-sec', type=float, default=30.0,
                   help='Padding around each event when matching windows')

    # Rule thresholds, provisional; the review sheet is where labels get confirmed.
    p.add_argument('--rule-rms-factor', type=float, default=3.0,
                   help='Positive when rms >= factor x reference rms')
    p.add_argument('--reference-rms', type=float, default=None,
                   help='Reference rms (counts); default: 10th percentile of this record')
    p.add_argument('--rule-band', type=float, nargs=2, default=(5.0, 30.0), metavar=('LO', 'HI'),
                   help='Band for the diagnostic band_ratio / STA/LTA / kurtosis columns')

    p.add_argument('--burst-factor', type=float, default=20.0,
                   help='timeofday: night window with raw peak >= factor x station median night rms -> class '
                        '(single vehicle pass); 0 disables')
    p.add_argument('--exclude-events', action='store_true',
                   help='Drop windows overlapping catalogued earthquakes (USGS) in the range')
    p.add_argument('--event-minmag', type=float, default=2.0)
    p.add_argument('--event-radius-deg', type=float, default=2.0)
    p.add_argument('--event-coda-sec', type=float, default=120.0,
                   help='Exclusion runs from origin time to origin + this many seconds')
    p.add_argument('--tz', default='America/Anchorage', help='Local time zone for --label-from timeofday')
    p.add_argument('--day-hours', type=int, nargs=2, default=(7, 19), metavar=('H0', 'H1'),
                   help='Local hours [H0, H1) labeled --class-name')
    p.add_argument('--night-hours', type=int, nargs=2, default=(1, 5), metavar=('H0', 'H1'),
                   help='Local hours [H0, H1) labeled Noise')

    p.add_argument('--freqmin', type=float, default=2.0, help='Training bandpass low (AK convention 2 Hz)')
    p.add_argument('--freqmax', type=float, default=20.0, help='Training bandpass high (AK convention 20 Hz)')

    p.add_argument('--out', default=str(REPO_ROOT / 'notebooks' / '02_labeling' / 'labeled_data'))
    p.add_argument('--prefix', default=None, help='File prefix (default <network>_<class-name lower>)')
    p.add_argument('--review-sheet', type=int, default=0,
                   help='Write a PNG grid + CSV of N random positive windows for eye-check')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not args.config and not (args.network and args.station and args.start and args.end):
        p.error('give --config, or all of --network --station --start --end')
    return args


def runs_from_config(args):
    """Expand a station config into one argparse namespace per collection day."""
    with open(args.config, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    col = cfg['collection']
    roles = set(col.get('roles', ['primary']))
    stations = [st['code'] for st in cfg['stations'] if st.get('role', 'primary') in roles]
    if not stations:
        raise SystemExit(f'No stations with role in {sorted(roles)} in {args.config}')
    tz = ZoneInfo(col.get('tz', args.tz))
    utc = ZoneInfo('UTC')
    runs = []
    for day in col['days']:
        day = str(day)[:10]
        # A config day is a local calendar day (midnight to midnight in col.tz),
        # so the weekday/weekend split and the {date} prefix mean what they say.
        local0 = datetime.fromisoformat(day).replace(tzinfo=tz)
        local1 = (local0 + timedelta(days=1)).replace(tzinfo=None).replace(tzinfo=tz)  # DST-safe next midnight
        t0, t1 = UTCDateTime(local0.astimezone(utc)), UTCDateTime(local1.astimezone(utc))
        run = argparse.Namespace(**vars(args))
        run.network = col.get('network', 'AM')
        run.station = stations
        run.start, run.end = str(t0), str(t1)
        run.class_name = col.get('class_name', 'Traffic')
        run.label_from = col.get('label_from', 'timeofday')
        # Any collection key that names a flag overrides it, so the YAML can
        # carry every setting that changes the output (reproducible pull).
        for key in ('channel', 'chunk_hours', 'events', 'event_pad_sec', 'rule_rms_factor', 'reference_rms',
                    'rule_band', 'tz', 'day_hours', 'night_hours', 'burst_factor', 'exclude_events',
                    'event_minmag', 'event_radius_deg', 'event_coda_sec', 'freqmin', 'freqmax',
                    'review_sheet', 'seed'):
            if key in col:
                value = col[key]
                setattr(run, key, tuple(value) if isinstance(value, list) else value)
        run.prefix = col.get('prefix_pattern', '{network}_{class}_{date}').format(
            network=run.network, **{'class': run.class_name.lower()}, date=day)
        runs.append(run)
    return runs


def with_retries(fn, what, attempts=3):
    """Call fn(); on any exception wait and retry (the Raspberry Shake server 502s now and then)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as err:
            if attempt == attempts:
                raise
            print(f'  {what}: {type(err).__name__} (attempt {attempt}/{attempts}), retrying')
            time.sleep(15 * attempt)


def pick_vertical_channel(client, network, station, t0, t1, requested=None):
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


def fetch_stream(client, network, station, channel, t0, t1, chunk_hours, native_sr=100.0, min_fraction=0.95):
    """
    Download in chunks, return one merged Stream (gaps left as separate traces).

    A chunk that comes back with fewer than ``min_fraction`` of the expected
    samples is treated as a failed attempt and re-requested (the Raspberry
    Shake server returns truncated responses under load); the longest
    response of the attempts is kept.
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
                st = client.get_waveforms(network, station, '*', channel, t - 5, te + 5)
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
    out.merge(method=1, fill_value=None)  # gaps become masked
    return out.split()                    # ...and then separate gap-free traces


def preprocess(tr, freqmin, freqmax):
    """AK-notebook preprocessing on one gap-free trace, returns a copy."""
    tr = tr.copy()
    tr.detrend('linear')
    tr.detrend('demean')
    # Fixed 2 s taper (a percentage of a day-long trace would eat whole windows);
    # windows within EDGE_SEC of a segment edge are skipped in window_starts.
    tr.taper(max_percentage=None, max_length=EDGE_SEC)
    if tr.stats.sampling_rate != SAMPLING_RATE:
        tr.resample(SAMPLING_RATE)
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4)
    return tr


def rule_features(tr_raw, band):
    """Per-window traffic-band features on a resampled, detrended, unfiltered trace."""
    tr_band = tr_raw.copy().filter('bandpass', freqmin=band[0], freqmax=band[1], corners=4)
    sr = tr_raw.stats.sampling_rate
    cft = classic_sta_lta(tr_band.data, int(1.0 * sr), int(10.0 * sr))
    return tr_band.data, cft


def window_starts(tr, global_t0):
    """Window start samples aligned to whole minutes counted from global_t0."""
    sr = tr.stats.sampling_rate
    offset = (tr.stats.starttime - global_t0) % WINDOW_SEC
    tol = 0.5 / sr  # half a sample: float time arithmetic can leave 1e-9 s remainders
    first = 0 if (offset < tol or WINDOW_SEC - offset < tol) else int(round((WINDOW_SEC - offset) * sr))
    n = len(tr.data)
    edge = int(EDGE_SEC * sr)
    starts = range(first, n - WINDOW_SAMPLES + 1, WINDOW_SAMPLES)
    # Keep clear of the taper / filter transient at both ends of the segment
    return [s for s in starts if s >= edge and s + WINDOW_SAMPLES <= n - edge]


def load_events(path):
    ev = pd.read_csv(path)
    if 'time' not in ev.columns:
        raise ValueError(f'{path} needs a "time" column (UTC)')
    ev['t0'] = ev['time'].apply(UTCDateTime)
    ev['t1'] = ev['t0'] + ev.get('duration_sec', pd.Series(0.0, index=ev.index)).fillna(0.0)
    if 'event_id' not in ev.columns:
        ev['event_id'] = [f'ev{i:05d}' for i in range(len(ev))]
    return ev


def catalog_events(t0, t1, lat, lon, minmag, radius_deg):
    """USGS events in the range; returns [(origin UTCDateTime, mag, dist_km)]."""
    from obspy.geodetics import gps2dist_azimuth
    try:
        cat = Client('USGS', timeout=60).get_events(starttime=t0, endtime=t1, latitude=lat, longitude=lon,
                                                     maxradius=radius_deg, minmagnitude=minmag)
    except Exception as err:  # no events (204) or catalog down: nothing to exclude
        print(f'  catalog query: {type(err).__name__}, no exclusions')
        return []
    out = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or ev.magnitudes[0]
        out.append((o.time, m.mag, gps2dist_azimuth(lat, lon, o.latitude, o.longitude)[0] / 1000))
    return out


def overlapping_event(ev, w0, w1, pad):
    if ev is None:
        return None
    hit = ev[(ev['t1'] + pad >= w0) & (ev['t0'] - pad <= w1)]
    return None if hit.empty else str(hit.iloc[0]['event_id'])


def main():
    args = parse_args()
    if args.config:
        for run in runs_from_config(args):
            print(f'\n=== {run.prefix}: {run.network} {" ".join(run.station)} {run.start[:10]} ===')
            collect(run)
    else:
        collect(args)


def collect(args):
    rng = np.random.default_rng(args.seed)
    t0, t1 = UTCDateTime(args.start), UTCDateTime(args.end)
    if t1 <= t0:
        raise SystemExit('--end must be after --start')
    positive_label = NAME_TO_LABEL[args.class_name.lower()]
    prefix = args.prefix or f'{args.network}_{args.class_name.lower()}'
    events = load_events(args.events) if args.label_from == 'events' else None
    tz = ZoneInfo(args.tz)
    if args.label_from == 'events' and events is None:
        raise SystemExit('--label-from events needs --events')

    fdsn = args.fdsn or FDSN_SERVERS.get(args.network, 'IRIS')
    client = Client(base_url=fdsn, timeout=120) if fdsn.startswith('http') else Client(fdsn, timeout=120)
    print(f'FDSN: {fdsn}')

    windows, rows = [], []
    for station in args.station:
        print(f'\n{args.network}.{station}')
        channel, (lat, lon, native_sr) = pick_vertical_channel(client, args.network, station, t0, t1, args.channel)
        print(f'  channel {channel} @ {native_sr:g} Hz, lat {lat:.4f} lon {lon:.4f}')
        traces = fetch_stream(client, args.network, station, channel, t0, t1, args.chunk_hours, native_sr)
        quakes = []
        if args.exclude_events:
            quakes = catalog_events(t0, t1, lat, lon, args.event_minmag, args.event_radius_deg)
            print(f'  excluding windows overlapping {len(quakes)} catalogued events '
                  f'(M>={args.event_minmag}, <={args.event_radius_deg} deg): '
                  + ', '.join(f'M{m:.1f}@{d:.0f}km' for _, m, d in sorted(quakes, key=lambda q: q[2])[:6]))
        n_quake_dropped = 0
        for tr in traces:
            if len(tr.data) < WINDOW_SAMPLES * (tr.stats.sampling_rate / SAMPLING_RATE) + 2:
                continue
            tr_raw = tr.copy()
            tr_raw.data = tr_raw.data.astype(np.float64)  # miniSEED ints would overflow in x**2
            tr_raw.detrend('linear').detrend('demean')
            if tr_raw.stats.sampling_rate != SAMPLING_RATE:
                tr_raw.resample(SAMPLING_RATE)
            tr_train = preprocess(tr, args.freqmin, args.freqmax)
            band_data, cft = rule_features(tr_raw, args.rule_band)
            sr = tr_train.stats.sampling_rate
            for s in window_starts(tr_train, t0):
                e = s + WINDOW_SAMPLES
                w_train = tr_train.data[s:e].astype(np.float32)
                w_raw = np.asarray(tr_raw.data[s:e], dtype=np.float64)
                w_band = np.asarray(band_data[s:e], dtype=np.float64)
                w_t0 = tr_train.stats.starttime + s / sr
                w_t1 = w_t0 + WINDOW_SEC
                # Half-sample tolerance: Raspberry Shake samples sit ~2 ms off the whole second
                if w_t0 < t0 - 0.5 / sr or w_t1 > t1 + 0.5 / sr or not np.all(np.isfinite(w_train)):
                    continue
                if any(q_t <= w_t1 and q_t + args.event_coda_sec >= w_t0 for q_t, _, _ in quakes):
                    n_quake_dropped += 1
                    continue
                total = float(np.sum(w_raw ** 2)) or 1.0
                feats = {
                    'rms': float(np.sqrt(np.mean(w_raw ** 2))),
                    'peak': float(np.max(np.abs(w_raw))),
                    'band_ratio': float(np.sum(w_band ** 2) / total),
                    'stalta_band_max': float(np.max(cft[s:e])),
                    'kurtosis_band': float(kurtosis(w_band)),
                }
                event_id = None
                if args.label_from == 'rule':
                    positive = None  # decided below once the reference rms is known
                    method = 'rule:rms'
                elif args.label_from == 'timeofday':
                    hour = w_t0.datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(tz).hour
                    if args.day_hours[0] <= hour < args.day_hours[1]:
                        positive = True
                    elif args.night_hours[0] <= hour < args.night_hours[1]:
                        positive = False
                    else:
                        continue  # ambiguous hours are not used
                    method = f'timeofday:{args.tz}:day{args.day_hours[0]}-{args.day_hours[1]}:night{args.night_hours[0]}-{args.night_hours[1]}'
                elif args.label_from == 'events':
                    event_id = overlapping_event(events, w_t0, w_t1, args.event_pad_sec)
                    positive = event_id is not None
                    method = f'events:{Path(args.events).name}'
                else:
                    positive, method = True, 'all'
                label = positive_label if positive else 0
                windows.append(w_train)
                rows.append({
                    'window_id': len(rows),
                    'event_id': event_id,
                    'network': args.network,
                    'station': station,
                    'location': tr.stats.location,
                    'channel': channel,
                    'latitude': lat,
                    'longitude': lon,
                    'start_time': str(w_t0),
                    'end_time': str(w_t1),
                    'sampling_rate': SAMPLING_RATE,
                    'native_sampling_rate': float(tr.stats.sampling_rate),
                    'window_len_sec': WINDOW_SEC,
                    'p_offset_sec': None,
                    'window_type': LABEL_MAP[label].lower(),
                    'label': label,
                    'label_name': LABEL_MAP[label],
                    'label_method': method,
                    'source': args.network,
                    'bandpass_hz': f'{args.freqmin}-{args.freqmax}',
                    **feats,
                    'reviewed': '',
                    'review_label': '',
                })

        if args.exclude_events:
            print(f'  dropped {n_quake_dropped} windows overlapping catalogued events')

    if not windows:
        raise SystemExit('No complete windows; check station, channel and time range.')

    X = np.stack(windows)
    meta = pd.DataFrame(rows)
    if args.label_from == 'timeofday' and args.burst_factor > 0:
        night = meta['label'] == 0
        ref = meta[night].groupby('station')['rms'].median()
        meta['night_rms_ref'] = meta['station'].map(ref)
        meta['burst_ratio'] = meta['peak'] / meta['night_rms_ref']
        burst = night & (meta['burst_ratio'] >= args.burst_factor)
        meta.loc[burst, 'label'] = positive_label
        meta.loc[burst, 'label_name'] = LABEL_MAP[positive_label]
        meta.loc[burst, 'window_type'] = LABEL_MAP[positive_label].lower()
        meta.loc[burst, 'label_method'] = meta.loc[burst, 'label_method'] + f'+burst>={args.burst_factor:g}x'
        print(f'\nNight windows relabeled {LABEL_MAP[positive_label]} as single-vehicle bursts '
              f'(peak >= {args.burst_factor:g} x station night median rms): '
              + ', '.join(f'{k}={int(v)}' for k, v in burst.groupby(meta['station']).sum().items()))
    if args.label_from == 'rule':
        # Station-relative: one reference per station (site gain and local noise
        # differ), unless a single --reference-rms is given for all of them.
        if args.reference_rms:
            meta['reference_rms'] = float(args.reference_rms)
        else:
            meta['reference_rms'] = meta.groupby('station')['rms'].transform(lambda r: np.percentile(r, 10))
        positive = meta['rms'] >= args.rule_rms_factor * meta['reference_rms']
        meta['label'] = np.where(positive, positive_label, 0)
        meta['label_name'] = meta['label'].map(LABEL_MAP)
        meta['window_type'] = meta['label_name'].str.lower()
        meta['label_method'] = f'rule:rms>={args.rule_rms_factor}x' + (
            f'{args.reference_rms:.1f}' if args.reference_rms else 'station_p10')
        meta['rms_ratio'] = meta['rms'] / meta['reference_rms']
        refs = meta.groupby('station')['reference_rms'].first()
        print(f'\nRule factor {args.rule_rms_factor}, reference rms per station '
              f'({"given" if args.reference_rms else "10th percentile"}): '
              + ', '.join(f'{k}={v:.1f}' for k, v in refs.items()))
    y = meta['label'].to_numpy(dtype=np.int64)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wf, lf, mf = (out / f'{prefix}_waveforms_{stamp}.npy', out / f'{prefix}_labels_{stamp}.npy',
                  out / f'{prefix}_metadata_{stamp}.csv')
    np.save(wf, X)
    np.save(lf, y)
    meta.to_csv(mf, index=False)

    counts = meta.groupby(['station', 'label_name']).size().unstack(fill_value=0)
    summary = out / f'{prefix}_summary_{stamp}.txt'
    with open(summary, 'w') as f:
        f.write(f'Continuous-window dataset ({prefix})\n{"=" * 60}\n\n')
        f.write(f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        f.write(f'Command: {" ".join(sys.argv)}\n\n')
        f.write(f'Range: {t0} -> {t1}\nWindow: {WINDOW_SEC:.0f} s @ {SAMPLING_RATE:g} Hz, no overlap, '
                f'bandpass {args.freqmin}-{args.freqmax} Hz\n')
        f.write(f'Positive class: {args.class_name} (label {positive_label}), method {args.label_from}\n\n')
        f.write('Windows per station and label:\n')
        f.write(counts.to_string() + '\n\n')
        f.write(f'Files:\n  {wf.name}  {X.shape} {X.dtype}\n  {lf.name}\n  {mf.name}\n')
    print(f'\nSaved {len(y)} windows -> {wf.name}\n{counts.to_string()}')

    if args.review_sheet:
        write_review_sheet(X, meta, positive_label, args.review_sheet, out / f'{prefix}_review_{stamp}', rng)


def write_review_sheet(X, meta, positive_label, n, stem, rng):
    """PNG grid (waveform + spectrogram) and a CSV with a blank keep column."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    pos = np.flatnonzero(meta['label'].to_numpy() == positive_label)
    if len(pos) == 0:
        print('Review sheet: no positive windows to show')
        return
    pick = np.sort(rng.choice(pos, size=min(n, len(pos)), replace=False))
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(4.2 * ncol, 2.4 * nrow),
                             gridspec_kw={'height_ratios': [1, 1.4] * nrow,
                                          'hspace': 0.45, 'wspace': 0.3}, squeeze=False)
    t = np.arange(WINDOW_SAMPLES) / SAMPLING_RATE
    for k, idx in enumerate(pick):
        r, c = divmod(k, ncol)
        ax_w, ax_s = axes[2 * r, c], axes[2 * r + 1, c]
        w = X[idx]
        ax_w.plot(t, w, lw=0.4, color='#333')
        ax_w.set_xlim(0, WINDOW_SEC)
        ax_w.set_xticks([])
        ax_w.tick_params(labelsize=6)
        m = meta.iloc[idx]
        ax_w.set_title(f'#{idx} {m.station} {m.start_time[:19]}\n'
                       f'rms {m.rms:.0f} ratio {m.band_ratio:.2f} stalta {m.stalta_band_max:.1f} '
                       f'kurt {m.kurtosis_band:.1f}', fontsize=7)
        f_, t_, S = spectrogram(w, fs=SAMPLING_RATE, nperseg=256, noverlap=192)
        S_db = 10 * np.log10(S + 1e-12)
        ax_s.pcolormesh(t_, f_, S_db, shading='auto', cmap='magma',
                        vmin=np.percentile(S_db, 5), vmax=np.percentile(S_db, 99))
        ax_s.set_ylim(0, 50)
        ax_s.set_xlim(0, WINDOW_SEC)
        ax_s.tick_params(labelsize=6)
        if r == nrow - 1:
            ax_s.set_xlabel('s', fontsize=7)
        else:
            ax_s.set_xticklabels([])
    for k in range(len(pick), nrow * ncol):
        r, c = divmod(k, ncol)
        axes[2 * r, c].axis('off')
        axes[2 * r + 1, c].axis('off')
    fig.suptitle(f'Provisional {LABEL_MAP[positive_label]} windows: mark keep=1/0 in {stem.name}.csv', fontsize=9)
    fig.savefig(f'{stem}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    sheet = meta.loc[pick, ['window_id', 'station', 'start_time', 'label_name', 'label_method', 'rms',
                            'band_ratio', 'stalta_band_max', 'kurtosis_band']].copy()
    sheet['keep'] = ''
    sheet['note'] = ''
    sheet.to_csv(f'{stem}.csv', index=False)
    print(f'Review sheet: {stem}.png and {stem}.csv ({len(pick)} windows)')


if __name__ == '__main__':
    main()
