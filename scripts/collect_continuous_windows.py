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
  rule       station-relative energy rule: a window is positive when its
             5-30 Hz rms is at least ``--rule-rms-factor`` times a reference
             (the record's 10th percentile, or ``--reference-rms`` from a
             quiet run of the same station). The band rms, not the raw rms:
             AK strong-motion accelerometers carry sub-5 Hz self-noise that
             dominates the raw value. On AM.R4017 the daytime median rms
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
  events     a CSV of event times (``--events``, columns ``time`` UTC and
             optional ``duration_sec``, ``event_id``, ``station``; rows with a
             station apply to that station only).
             With ``--event-search-sec 0`` windows overlapping an event get
             ``--class-name``, others Noise. With a search span (default
             900 s, for timetables and ADS-B times that are only good to
             minutes) the windows whose rms reaches ``--event-detect-factor``
             times the median of the span are the event; the rest of the
             span is dropped as ambiguous; windows outside every span are
             Noise. Per-station timing offsets come from
             ``--event-offsets`` (``STA=+9,STA2=-3`` minutes) so one
             timetable serves stations along a line (#10, #11).
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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta
from scipy.stats import kurtosis

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# src.data imports without torch (the PyTorch Dataset is loaded lazily).
from src.data.labels import LABEL_MAP, NAME_TO_LABEL
from src.data.collect import (
    BANDPASS, SAMPLING_RATE, WINDOW_SAMPLES, WINDOW_SEC,
    catalog_events, detrend_resample, fetch_stream, local_day_bounds, local_hour, make_client,
    label_event_detections, overlaps_event, pick_vertical_channel, preprocess, relabel_bursts, window_starts,
    write_dataset,
)


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
                   help='Padding around each event when matching windows (--event-search-sec 0)')
    p.add_argument('--event-search-sec', type=float, default=900.0,
                   help='events: search +/- this many seconds around each event time and keep the windows '
                        'whose rms stands out; 0 = plain overlap with --event-pad-sec')
    p.add_argument('--event-detect-factor', type=float, default=3.0,
                   help='events: rms must reach factor x the median rms of the search span')
    p.add_argument('--noise-keep', type=float, default=1.0,
                   help='keep this fraction of the Noise windows (seeded); a season of one station is mostly Noise')
    p.add_argument('--positive-keep', type=float, default=1.0,
                   help='keep this fraction of the positive windows (seeded); daytime Traffic is abundant')
    p.add_argument('--noise-hours', type=int, nargs=2, default=None, metavar=('H0', 'H1'),
                   help='keep Noise only in local hours [H0, H1): makes Noise mean the same thing in every set '
                        '(timeofday mode already does this through --night-hours)')
    p.add_argument('--noise-max-ratio', type=float, default=None,
                   help='drop Noise windows whose band rms exceeds this multiple of the station rolling median '
                        '(+/- --rule-local-min): unscheduled trains and other unlabeled events stay out of Noise')
    p.add_argument('--event-offsets', default=None,
                   help='events: per-station minutes added to event times, e.g. "K208=+9,R1796=+8,K222=+30"')

    # Rule thresholds, provisional; the review sheet is where labels get confirmed.
    p.add_argument('--rule-rms-factor', type=float, default=3.0,
                   help='Positive when rms >= factor x reference rms')
    p.add_argument('--reference-rms', type=float, default=None,
                   help='Reference rms (counts); default: 10th percentile of this record')
    p.add_argument('--rule-band', type=float, nargs=2, default=(5.0, 30.0), metavar=('LO', 'HI'),
                   help='Band for rms_band / peak_band / band_ratio / STA/LTA / kurtosis (5-30 Hz for traffic and '
                        'trains; 20-45 Hz for aircraft, whose jet noise sits above the road-traffic band)')
    p.add_argument('--rule-reference', default='station_p10', choices=['station_p10', 'local'],
                   help="rule: reference rms per station = 10th percentile of the record (station_p10, a quiet "
                        "floor; right for sustained daytime traffic) or the rolling median over +/- "
                        "--rule-local-min minutes (local; right for 1-2 min excursions such as aircraft)")
    p.add_argument('--rule-local-min', type=float, default=15.0)
    p.add_argument('--rule-coincidence', type=int, default=1,
                   help='rule: keep a positive only if at least this many stations have a positive within +/- 1 min '
                        '(an aircraft is heard at every station under its path; a car at one); the others are dropped')
    p.add_argument('--rule-min-active-sec', type=float, default=0.0,
                   help='rule: also require this many seconds of the window with 1 s band rms >= factor x reference '
                        '(an aircraft pass is a 60-90 s broadband burst; a car is 3-6 s)')

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

    p.add_argument('--freqmin', type=float, default=BANDPASS[0], help='Training bandpass low (AK convention 2 Hz)')
    p.add_argument('--freqmax', type=float, default=BANDPASS[1], help='Training bandpass high (AK convention 20 Hz)')

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
    tz = col.get('tz', args.tz)
    runs = []
    days = []
    for item in col['days']:
        if isinstance(item, dict):  # {start: ..., end: ...} inclusive range of local days
            d0, d1 = (datetime.fromisoformat(str(item[k])[:10]).date() for k in ('start', 'end'))
            days += [str(d0 + timedelta(days=i)) for i in range((d1 - d0).days + 1)]
        else:
            days.append(str(item)[:10])
    for day in days:
        # A config day is a local calendar day (midnight to midnight in col.tz),
        # so the weekday/weekend split and the {date} prefix mean what they say.
        t0, t1 = local_day_bounds(day, tz)
        run = argparse.Namespace(**vars(args))
        run.network = col.get('network', 'AM')
        run.station = stations
        run.start, run.end = str(t0), str(t1)
        run.class_name = col.get('class_name', 'Traffic')
        run.label_from = col.get('label_from', 'timeofday')
        # Any collection key that names a flag overrides it, so the YAML can
        # carry every setting that changes the output (reproducible pull).
        for key in ('channel', 'chunk_hours', 'events', 'event_pad_sec', 'event_search_sec', 'event_detect_factor',
                    'event_offsets', 'noise_keep', 'positive_keep', 'noise_hours', 'noise_max_ratio',
                    'rule_rms_factor', 'rule_reference', 'rule_local_min',
                    'rule_min_active_sec', 'rule_coincidence', 'reference_rms',
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


def rule_features(tr_raw, band):
    """Traffic-band copy and STA/LTA (1 s / 10 s) of a detrended, resampled, unfiltered trace."""
    tr_band = tr_raw.copy().filter('bandpass', freqmin=band[0], freqmax=band[1], corners=4)
    sr = tr_raw.stats.sampling_rate
    cft = classic_sta_lta(tr_band.data, int(1.0 * sr), int(10.0 * sr))
    return tr_band.data, cft


def parse_offsets(spec):
    """'K208=+9,R1796=+8' -> {'K208': 9.0, 'R1796': 8.0} (minutes); accepts a dict from YAML too."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    out = {}
    for item in str(spec).split(','):
        if item.strip():
            k, v = item.split('=')
            out[k.strip()] = float(v)
    return out


def load_events(path):
    ev = pd.read_csv(path)
    if 'time' not in ev.columns:
        raise ValueError(f'{path} needs a "time" column (UTC)')
    ev['t0'] = ev['time'].apply(UTCDateTime)
    ev['t1'] = ev['t0'] + ev.get('duration_sec', pd.Series(0.0, index=ev.index)).fillna(0.0)
    if 'event_id' not in ev.columns:
        ev['event_id'] = [f'ev{i:05d}' for i in range(len(ev))]
    return ev


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
    if args.label_from == 'events' and args.events and not Path(args.events).is_absolute():
        args.events = str(REPO_ROOT / args.events) if not Path(args.events).exists() else args.events
    events = load_events(args.events) if args.label_from == 'events' else None
    tz = args.tz
    if args.label_from == 'events' and events is None:
        raise SystemExit('--label-from events needs --events')

    client = make_client(args.network, args.fdsn)
    print(f'FDSN: {client.base_url}')

    windows, rows, band_secs = [], [], []   # band_secs: 1 s rms of the rule band per window, for the duration test
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
            tr_raw = detrend_resample(tr)
            tr_train = preprocess(tr, args.freqmin, args.freqmax)
            band_data, cft = rule_features(tr_raw, args.rule_band)
            sr = tr_train.stats.sampling_rate
            for s in window_starts(len(tr_train.data), sr, float(tr_train.stats.starttime - t0)):
                e = s + WINDOW_SAMPLES
                w_train = tr_train.data[s:e].astype(np.float32)
                w_raw = np.asarray(tr_raw.data[s:e], dtype=np.float64)
                w_band = np.asarray(band_data[s:e], dtype=np.float64)
                w_t0 = tr_train.stats.starttime + s / sr
                w_t1 = w_t0 + WINDOW_SEC
                # Half-sample tolerance: Raspberry Shake samples sit ~2 ms off the whole second
                if w_t0 < t0 - 0.5 / sr or w_t1 > t1 + 0.5 / sr or not np.all(np.isfinite(w_train)):
                    continue
                if overlaps_event(w_t0, w_t1, quakes, args.event_coda_sec):
                    n_quake_dropped += 1
                    continue
                total = float(np.sum(w_raw ** 2)) or 1.0
                feats = {
                    'rms': float(np.sqrt(np.mean(w_raw ** 2))),
                    'peak': float(np.max(np.abs(w_raw))),
                    'rms_band': float(np.sqrt(np.mean(w_band ** 2))),   # 5-30 Hz: what the rule, burst and
                    'peak_band': float(np.max(np.abs(w_band))),         # event detection use (accelerometers
                    'band_ratio': float(np.sum(w_band ** 2) / total),    # carry strong sub-5 Hz self-noise)
                    'stalta_band_max': float(np.max(cft[s:e])),
                    'kurtosis_band': float(kurtosis(w_band)),
                }
                event_id = None
                if args.label_from == 'rule':
                    positive = None  # decided below once the reference rms is known
                    method = 'rule:rms'
                elif args.label_from == 'timeofday':
                    hour = int(local_hour(w_t0, tz))
                    if args.day_hours[0] <= hour < args.day_hours[1]:
                        positive = True
                    elif args.night_hours[0] <= hour < args.night_hours[1]:
                        positive = False
                    else:
                        continue  # ambiguous hours are not used
                    method = f'timeofday:{args.tz}:day{args.day_hours[0]}-{args.day_hours[1]}:night{args.night_hours[0]}-{args.night_hours[1]}'
                elif args.label_from == 'events' and args.event_search_sec > 0:
                    positive = False  # decided by label_event_detections once all windows are in
                    method = f'events:{Path(args.events).name}'
                elif args.label_from == 'events':
                    event_id = overlapping_event(events, w_t0, w_t1, args.event_pad_sec)
                    positive = event_id is not None
                    method = f'events:{Path(args.events).name}'
                else:
                    positive, method = True, 'all'
                label = positive_label if positive else 0
                windows.append(w_train)
                band_secs.append(np.sqrt(np.mean(w_band[:int(WINDOW_SEC) * int(sr)].reshape(int(WINDOW_SEC), -1) ** 2, axis=1)))
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
    if args.label_from == 'events' and args.event_search_sec > 0:
        offsets = parse_offsets(args.event_offsets)
        parts, reports = [], []
        for station, sub in meta.groupby('station', sort=False):
            if 'station' in events.columns:  # rows with a blank station apply to every station
                st = events['station'].fillna('').astype(str).str.strip()
                ev = events[(st == station) | (st == '')].copy()
            else:
                ev = events.copy()
            ev['t0'] = ev['t0'] + 60.0 * offsets.get(station, 0.0)
            ev = ev[(ev['t0'] >= t0 - args.event_search_sec) & (ev['t0'] <= t1 + args.event_search_sec)]
            ev = ev.sort_values('t0', kind='stable')  # earlier event wins an overlap, whatever the CSV order
            labeled, rep = label_event_detections(sub, ev, positive_label, args.event_search_sec,
                                                  args.event_detect_factor)
            parts.append(labeled)
            reports.append(rep)
        meta = pd.concat(parts).sort_index()
        report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
        keep = ~meta['drop'].to_numpy()
        n_ambiguous = int((~keep).sum())
        X = X[keep]
        meta = meta[keep].drop(columns=['drop']).reset_index(drop=True)
        meta['window_id'] = np.arange(len(meta))
        if len(report):
            print(f'\nScheduled events: {int(report.detected.sum())} of {len(report)} detected '
                  f'(rms >= {args.event_detect_factor:g} x span median within +/- {args.event_search_sec:g} s); '
                  f'{n_ambiguous} ambiguous windows dropped')
            print(report.to_string(index=False))
            Path(args.out).mkdir(parents=True, exist_ok=True)
            report.to_csv(Path(args.out) / f'{prefix}_events_report.csv', index=False)
    if args.label_from == 'timeofday' and args.burst_factor > 0:
        counts_burst = relabel_bursts(meta, positive_label, args.burst_factor)
        print(f'\nNight windows relabeled {LABEL_MAP[positive_label]} as single-vehicle bursts '
              f'(peak >= {args.burst_factor:g} x station night median rms): '
              + ', '.join(f'{k}={v}' for k, v in counts_burst.items()))
    if args.label_from == 'rule':
        # Station-relative: one reference per station (site gain and local noise
        # differ), unless a single --reference-rms is given for all of them.
        if args.reference_rms:
            meta['reference_rms'] = float(args.reference_rms)
        elif args.rule_reference == 'local':
            k = int(args.rule_local_min)  # windows are one minute, so the rolling span is 2k + 1 windows
            meta['reference_rms'] = meta.groupby('station')['rms_band'].transform(
                lambda r: r.rolling(2 * k + 1, center=True, min_periods=k).median())
        else:
            meta['reference_rms'] = meta.groupby('station')['rms_band'].transform(lambda r: np.percentile(r, 10))
        env = np.stack(band_secs)  # (n_windows, 60) 1 s band rms
        meta['band_active_sec'] = (env >= args.rule_rms_factor * meta['reference_rms'].to_numpy()[:, None]).sum(axis=1)
        positive = (meta['rms_band'] >= args.rule_rms_factor * meta['reference_rms']) & \
                   (meta['band_active_sec'] >= args.rule_min_active_sec)
        if args.rule_coincidence > 1 and meta['station'].nunique() >= args.rule_coincidence:
            t = np.array([float(UTCDateTime(x)) for x in meta['start_time']])
            pos_by_station = {st: t[(meta['station'] == st).to_numpy() & positive.to_numpy()]
                              for st in meta['station'].unique()}
            n_coinc = np.zeros(len(meta), dtype=int)
            for i in np.flatnonzero(positive.to_numpy()):
                n_coinc[i] = sum(np.any(np.abs(tt - t[i]) <= 60.0) for tt in pos_by_station.values())
            meta['coincident_stations'] = n_coinc
            lonely = positive & (meta['coincident_stations'] < args.rule_coincidence)
            print(f'  coincidence: {int(lonely.sum())} single-station positives dropped, '
                  f'{int((positive & ~lonely).sum())} kept')
            keep = ~lonely.to_numpy()
            X, meta, positive = X[keep], meta[keep].reset_index(drop=True), positive[keep].reset_index(drop=True)
            meta['window_id'] = np.arange(len(meta))
        meta['label'] = np.where(positive, positive_label, 0)
        meta['label_name'] = meta['label'].map(LABEL_MAP)
        meta['window_type'] = meta['label_name'].str.lower()
        meta['label_method'] = f'rule:rms{args.rule_band[0]:g}-{args.rule_band[1]:g}Hz>={args.rule_rms_factor}x' + (
            f'{args.reference_rms:.1f}' if args.reference_rms else
            (f'local{args.rule_local_min:g}min' if args.rule_reference == 'local' else 'station_p10')) + (
            f',active>={args.rule_min_active_sec:g}s' if args.rule_min_active_sec > 0 else '') + (
            f',coincident>={args.rule_coincidence}' if args.rule_coincidence > 1 else '')
        meta['rms_ratio'] = meta['rms_band'] / meta['reference_rms']
        refs = meta.groupby('station')['reference_rms'].median()
        print(f'\nRule factor {args.rule_rms_factor}, {args.rule_band[0]:g}-{args.rule_band[1]:g} Hz, reference rms '
              f'per station ({"given" if args.reference_rms else args.rule_reference}, median shown): '
              + ', '.join(f'{k}={v:.1f}' for k, v in refs.items()))
    # ---- final pass, every mode: what Noise means, and how much of each class to keep
    keep = np.ones(len(meta), dtype=bool)
    is_noise = meta['label'].to_numpy() == 0
    if args.noise_hours is not None:
        hours = np.array([int(local_hour(UTCDateTime(x), tz)) for x in meta['start_time']])
        in_hours = (hours >= args.noise_hours[0]) & (hours < args.noise_hours[1])
        keep &= ~is_noise | in_hours
        print(f'\nNoise restricted to {args.noise_hours[0]:02d}-{args.noise_hours[1]:02d} local: '
              f'{int((is_noise & ~in_hours).sum())} Noise windows outside dropped')
    if args.noise_max_ratio is not None:
        k = int(args.rule_local_min)
        col = 'rms_band' if 'rms_band' in meta else 'rms'
        local_med = meta.groupby('station')[col].transform(lambda r: r.rolling(2 * k + 1, center=True, min_periods=k).median())
        loud = is_noise & (meta[col].to_numpy() > args.noise_max_ratio * local_med.to_numpy())
        keep &= ~loud
        print(f'Noise windows above {args.noise_max_ratio:g} x the local median dropped: {int(loud.sum())}')
    if args.noise_keep < 1.0:
        keep &= ~is_noise | (rng.random(len(meta)) < args.noise_keep)
    if args.positive_keep < 1.0:
        keep &= is_noise | (rng.random(len(meta)) < args.positive_keep)
    if not keep.all():
        X = X[keep]
        meta = meta[keep].reset_index(drop=True)
        meta['window_id'] = np.arange(len(meta))
        print(f'Kept {len(meta)} windows (noise_keep {args.noise_keep:g}, positive_keep {args.positive_keep:g})')
    y = meta['label'].to_numpy(dtype=np.int64)

    paths = write_dataset(args.out, prefix, X, y, meta, summary_lines=[
        f'Command: {" ".join(sys.argv)}',
        f'Range: {t0} -> {t1}',
        f'Window: {WINDOW_SEC:.0f} s @ {SAMPLING_RATE:g} Hz, no overlap, bandpass {args.freqmin}-{args.freqmax} Hz',
        f'Positive class: {args.class_name} (label {positive_label}), method {args.label_from}',
    ])
    counts = meta.groupby(['station', 'label_name']).size().unstack(fill_value=0)
    print(f'\nSaved {len(y)} windows -> {paths["waveforms"].name}\n{counts.to_string()}')
    stamp = paths['waveforms'].stem.split('_waveforms_')[-1]

    if args.review_sheet:
        write_review_sheet(X, meta, positive_label, args.review_sheet,
                           Path(args.out) / f'{prefix}_review_{stamp}', rng)


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
