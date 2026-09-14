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
    overlaps_event, pick_vertical_channel, preprocess, relabel_bursts, window_starts, write_dataset,
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
    for day in col['days']:
        day = str(day)[:10]
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


def rule_features(tr_raw, band):
    """Traffic-band copy and STA/LTA (1 s / 10 s) of a detrended, resampled, unfiltered trace."""
    tr_band = tr_raw.copy().filter('bandpass', freqmin=band[0], freqmax=band[1], corners=4)
    sr = tr_raw.stats.sampling_rate
    cft = classic_sta_lta(tr_band.data, int(1.0 * sr), int(10.0 * sr))
    return tr_band.data, cft


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
    events = load_events(args.events) if args.label_from == 'events' else None
    tz = args.tz
    if args.label_from == 'events' and events is None:
        raise SystemExit('--label-from events needs --events')

    client = make_client(args.network, args.fdsn)
    print(f'FDSN: {client.base_url}')

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
                    'band_ratio': float(np.sum(w_band ** 2) / total),
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
        counts_burst = relabel_bursts(meta, positive_label, args.burst_factor)
        print(f'\nNight windows relabeled {LABEL_MAP[positive_label]} as single-vehicle bursts '
              f'(peak >= {args.burst_factor:g} x station night median rms): '
              + ', '.join(f'{k}={v}' for k, v in counts_burst.items()))
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
