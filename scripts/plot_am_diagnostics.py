#!/usr/bin/env python3
"""
Diagnostic figures for a collected AM station-day: are the "Traffic" and
"Noise" windows what we think they are?

Re-fetches the raw day (unfiltered, so spectra cover 0-50 Hz) and draws:
  1. diurnal rms per minute for each station and day, with the labeled hours
     shaded and catalogued earthquakes marked
  2. Welch spectra of a day hour vs a night hour per station
  3. a 24 h spectrogram (one Welch PSD per minute) for one station-day
  4. example windows on a common amplitude scale: Noise, Traffic, and the
     window holding the nearest catalogued earthquake

Usage (from repo root):
  python scripts/plot_am_diagnostics.py --config configs/am_stations.yaml \
      --out docs/figures --cache /tmp/am_raw
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import welch

SR = 100.0
WIN = 60
NPERSEG = 1024


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/am_stations.yaml')
    p.add_argument('--out', default='docs/figures')
    p.add_argument('--cache', default=None, help='Directory for raw-day .npz caches (re-fetch if absent)')
    p.add_argument('--spectrogram-station', default=None, help='Station for figure 3 (default: first primary)')
    p.add_argument('--spectrogram-day', default=None, help='Day for figure 3 (default: first config day)')
    p.add_argument('--day-hour', type=int, default=10, help='Local hour for the "day" spectrum')
    p.add_argument('--night-hour', type=int, default=2, help='Local hour for the "night" spectrum')
    p.add_argument('--fdsn', default='https://data.raspberryshake.org')
    return p.parse_args()


def local_day_bounds(day, tz):
    t0 = datetime.fromisoformat(day).replace(tzinfo=tz)
    t1 = (t0 + timedelta(days=1)).replace(tzinfo=None).replace(tzinfo=tz)
    utc = ZoneInfo('UTC')
    return UTCDateTime(t0.astimezone(utc)), UTCDateTime(t1.astimezone(utc))


def fetch_raw_day(client, station, t0, t1, cache):
    """Full local day, merged with interpolated gaps, detrended, at 100 Hz; cached as npz."""
    path = cache / f'{station}_{t0.isoformat()[:10]}.npz' if cache else None
    if path and path.exists():
        z = np.load(path)
        return z['data'], UTCDateTime(str(z['start'])), (float(z['lat']), float(z['lon']))
    inv = client.get_stations(network='AM', station=station, level='station')
    lat, lon = inv[0][0].latitude, inv[0][0].longitude
    st = client.get_waveforms('AM', station, '*', 'EHZ', t0 - 5, t1 + 5)
    st.merge(method=1, fill_value='interpolate')
    tr = st[0]
    tr.data = tr.data.astype(np.float64)
    tr.detrend('linear').detrend('demean')
    if tr.stats.sampling_rate != SR:
        tr.resample(SR)
    tr.trim(t0, t1, pad=True, fill_value=0.0)
    if path:
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(path, data=tr.data, start=str(tr.stats.starttime), lat=lat, lon=lon)
    return tr.data, tr.stats.starttime, (lat, lon)


def minute_rms(data):
    n = len(data) // int(WIN * SR)
    x = data[:n * int(WIN * SR)].reshape(n, -1)
    return np.sqrt(np.mean(x ** 2, axis=1))


def minute_psd(data):
    n = len(data) // int(WIN * SR)
    x = data[:n * int(WIN * SR)].reshape(n, -1)
    f, P = welch(x, fs=SR, nperseg=NPERSEG, axis=1)
    return f, P


def nearby_events(t0, t1, lat, lon, maxradius_deg=2.0, minmag=2.0):
    try:
        cat = Client('USGS', timeout=60).get_events(starttime=t0, endtime=t1, latitude=lat, longitude=lon,
                                                     maxradius=maxradius_deg, minmagnitude=minmag)
    except Exception:
        return []
    out = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or ev.magnitudes[0]
        dist = gps2dist_azimuth(lat, lon, o.latitude, o.longitude)[0] / 1000
        out.append({'time': o.time, 'mag': m.mag, 'dist_km': dist})
    return sorted(out, key=lambda e: e['dist_km'])


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    col = cfg['collection']
    tz = ZoneInfo(col.get('tz', 'America/Anchorage'))
    roles = set(col.get('roles', ['primary']))
    stations = [s['code'] for s in cfg['stations'] if s.get('role', 'primary') in roles]
    days = [str(d)[:10] for d in col['days']]
    day_h, night_h = col.get('day_hours', [7, 19]), col.get('night_hours', [1, 5])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    client = Client(base_url=args.fdsn, timeout=120)

    raw, events = {}, {}
    for day in days:
        t0, t1 = local_day_bounds(day, tz)
        for sta in stations:
            print(f'fetching {sta} {day} ({t0.isoformat()[:16]}Z -> {t1.isoformat()[:16]}Z)')
            data, start, (lat, lon) = fetch_raw_day(client, sta, t0, t1, cache)
            raw[(sta, day)] = (data, start)
            if day not in events:
                events[day] = nearby_events(t0, t1, lat, lon)

    def local_hours(start, n, day):
        """Hours since local midnight of `day` for n consecutive minutes starting at `start`."""
        t0, _ = local_day_bounds(day, tz)
        return (float(start - t0) + np.arange(n) * WIN) / 3600.0

    def annotate(day):
        # nearest / largest only, so labels stay readable
        evs = [e for e in events.get(day, []) if e['dist_km'] <= 100 or e['mag'] >= 3.0]
        return evs[:4]

    def ev_hour(ev, day):
        t0, _ = local_day_bounds(day, tz)
        return float(ev['time'] - t0) / 3600.0

    # ---- Figure 1: diurnal rms, one panel per day, one line per station
    fig, axes = plt.subplots(1, len(days), figsize=(6.5 * len(days), 4), sharey=True, squeeze=False)
    for ax, day in zip(axes[0], days):
        for sta in stations:
            data, start = raw[(sta, day)]
            r = minute_rms(data)
            h = local_hours(start, len(r), day)
            ax.plot(h, r, lw=0.8, label=sta)
        ax.axvspan(day_h[0], day_h[1], color='tab:orange', alpha=0.10, label=f'{col.get("class_name", "Traffic")} hours')
        ax.axvspan(night_h[0], night_h[1], color='tab:blue', alpha=0.12, label='Noise hours')
        for ev in annotate(day):
            ax.axvline(ev_hour(ev, day), color='k', lw=0.7, ls='--')
            ax.text(ev_hour(ev, day), 1.5e4, f'M{ev["mag"]:.1f}\n{ev["dist_km"]:.0f} km', fontsize=7, ha='center', va='top')
        wd = datetime.fromisoformat(day).strftime('%A')
        ax.set_title(f'{day} ({wd}), rms per minute, raw counts')
        ax.set_xlabel(f'local hour ({tz.key})')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_yscale('log')
        ax.set_ylim(50, 2e4)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axes[0][0].set_ylabel('rms (counts)')
    fig.tight_layout()
    fig.savefig(out / 'am_diurnal_rms.png', dpi=130)
    plt.close(fig)

    # ---- Figure 2: day-hour vs night-hour spectra per station
    fig, axes = plt.subplots(1, len(stations), figsize=(6 * len(stations), 4.2), sharey=True, squeeze=False)
    for ax, sta in zip(axes[0], stations):
        for day in days:
            data, start = raw[(sta, day)]
            f, P = minute_psd(data)
            h = local_hours(start, P.shape[0], day)
            wd = datetime.fromisoformat(day).strftime('%a')
            for hour, name, ls in [(args.day_hour, 'day', '-'), (args.night_hour, 'night', '--')]:
                sel = (h >= hour) & (h < hour + 1)
                ax.semilogy(f, np.median(P[sel], axis=0), ls=ls, lw=1.2,
                            label=f'{wd} {hour:02d}-{hour + 1:02d} local ({name})')
        ax.axvspan(2, 20, color='grey', alpha=0.10, label='training band 2-20 Hz')
        ax.set_title(f'{sta}: median Welch PSD over one hour')
        ax.set_xlabel('frequency (Hz)')
        ax.set_xlim(0, 50)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel('PSD (counts$^2$/Hz)')
    fig.tight_layout()
    fig.savefig(out / 'am_day_night_psd.png', dpi=130)
    plt.close(fig)

    # ---- Figure 3: 24 h spectrogram for one station-day
    sta3 = args.spectrogram_station or stations[0]
    day3 = args.spectrogram_day or days[0]
    data, start = raw[(sta3, day3)]
    f, P = minute_psd(data)
    h = local_hours(start, P.shape[0], day3)
    S = 10 * np.log10(P.T + 1e-12)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.pcolormesh(h, f, S, shading='auto', cmap='magma',
                       vmin=np.percentile(S, 2), vmax=np.percentile(S, 99.5))
    for ev in annotate(day3):
        ax.axvline(ev_hour(ev, day3), color='w', lw=0.8, ls='--')
        ax.text(ev_hour(ev, day3), 48, f'M{ev["mag"]:.1f} {ev["dist_km"]:.0f} km', color='w',
                fontsize=7, ha='center', va='top')
    ax.axhline(2, color='w', lw=0.5, alpha=0.6)
    ax.axhline(20, color='w', lw=0.5, alpha=0.6)
    wd = datetime.fromisoformat(day3).strftime('%A')
    ax.set_title(f'{sta3} {day3} ({wd}): PSD per minute, dB rel. counts$^2$/Hz; lines at 2 and 20 Hz')
    ax.set_xlabel(f'local hour ({tz.key})')
    ax.set_ylabel('frequency (Hz)')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    fig.colorbar(im, ax=ax, pad=0.01, label='dB')
    fig.tight_layout()
    fig.savefig(out / f'am_spectrogram_{sta3}_{day3}.png', dpi=130)
    plt.close(fig)

    # ---- Figure 4: example windows on a common scale, from the stored (2-20 Hz) dataset
    labeled = Path('notebooks/02_labeling/labeled_data')
    prefix = col.get('prefix_pattern', 'AM_{class}_{date}').format(network='AM', date=day3,
                                                                    **{'class': col.get('class_name', 'Traffic').lower()})
    wfs = sorted(labeled.glob(f'{prefix}_waveforms_*.npy'))
    if wfs:
        stamp = wfs[-1].stem.split('_waveforms_')[-1]
        X = np.load(wfs[-1])
        meta = pd.read_csv(labeled / f'{prefix}_metadata_{stamp}.csv')
        meta['t0'] = meta.start_time.apply(UTCDateTime)
        sub = meta[meta.station == sta3]
        rng = np.random.default_rng(0)
        picks = []
        cls = col.get('class_name', 'Traffic')
        for name, how, k in (('Noise', 'timeofday', 3), (cls, 'timeofday', 2), (cls, 'burst', 2)):
            idx = sub.index[(sub.label_name == name) & (sub.label_method.str.contains(how) if how == 'burst'
                                                          else ~sub.label_method.str.contains('burst'))].to_numpy()
            picks += [(name, i) for i in rng.choice(idx, size=min(k, len(idx)), replace=False)]
        # Earthquake examples come from the raw day (the collector drops these
        # windows with --exclude-events), filtered like the stored windows.
        from obspy import Trace
        eq_panels = []
        for ev in events.get(day3, [])[:2]:
            tr = Trace(data=data.copy(), header={'sampling_rate': SR, 'starttime': start})
            tr.filter('bandpass', freqmin=2.0, freqmax=20.0, corners=4)
            i0 = int(np.floor(float(ev['time'] - start) / WIN) * WIN * SR)
            if 0 <= i0 and i0 + int(WIN * SR) <= len(tr.data):
                w = tr.data[i0:i0 + int(WIN * SR)].astype(np.float32)
                eq_panels.append((f'M{ev["mag"]:.1f} at {ev["dist_km"]:.0f} km (window excluded from training)',
                                  w, start + i0 / SR))
        n = len(picks) + len(eq_panels)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.3 * nrow), sharex=True, sharey=True, squeeze=False)
        t = np.arange(X.shape[1]) / SR
        ymax = max([np.abs(X[i]).max() for _, i in picks] + [np.abs(w).max() for _, w, _ in eq_panels])
        for k, (name, i) in enumerate(picks):
            ax = axes[k // ncol, k % ncol]
            ax.plot(t, X[i], lw=0.4, color='tab:orange' if name != 'Noise' else 'tab:blue')
            m = meta.loc[i]
            hl = UTCDateTime(m.start_time).datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)
            ax.set_title(f'{name} ({"burst" if "burst" in m.label_method else "timeofday"}), {hl:%H:%M} local, rms {m.rms:.0f}', fontsize=8)
            ax.grid(alpha=0.3)
        for j, (name, w, w_t0) in enumerate(eq_panels):
            k = len(picks) + j
            ax = axes[k // ncol, k % ncol]
            ax.plot(t, w, lw=0.4, color='tab:red')
            hl = w_t0.datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(tz)
            ax.set_title(f'{name}, {hl:%H:%M} local', fontsize=8)
            ax.grid(alpha=0.3)
        for k in range(n, nrow * ncol):
            axes[k // ncol, k % ncol].axis('off')
        axes[0, 0].set_ylim(-ymax, ymax)
        for ax in axes[-1]:
            ax.set_xlabel('s')
        fig.suptitle(f'{sta3} {day3}: stored 60 s windows (2-20 Hz), common amplitude scale', fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f'am_example_windows_{sta3}_{day3}.png', dpi=130)
        plt.close(fig)

    # ---- Figure 5: isolated transients in the quiet hours = single vehicle passes
    from obspy import Trace
    from scipy.signal import hilbert
    data, start = raw[(sta3, day3)]
    tr = Trace(data=data.copy(), header={'sampling_rate': SR, 'starttime': start})
    tr.filter('bandpass', freqmin=5.0, freqmax=30.0, corners=4)
    env = np.abs(hilbert(tr.data))
    env = pd.Series(env).rolling(int(0.5 * SR), center=True, min_periods=1).mean().to_numpy()
    hours = (float(start - local_day_bounds(day3, tz)[0]) + np.arange(len(env)) / SR) / 3600.0
    night = (hours >= night_h[0]) & (hours < night_h[1])
    base = np.median(env[night])
    above = night & (env > 6 * base)
    # group contiguous samples above threshold into bursts
    edges = np.flatnonzero(np.diff(above.astype(int)))
    starts_ = edges[::2] if above[0] == 0 else np.r_[0, edges[1::2]]
    ends_ = edges[1::2] if above[0] == 0 else edges[::2]
    bursts = []
    for a, b in zip(starts_, ends_):
        dur = (b - a) / SR
        if 2.0 <= dur <= 40.0:
            bursts.append((a, b, env[a:b].max() / base, dur))
    # keep bursts isolated from each other by at least 60 s, strongest first
    bursts.sort(key=lambda x: -x[2])
    kept = []
    for a, b, amp, dur in bursts:
        if all(abs(a - k[0]) > 60 * SR for k in kept):
            kept.append((a, b, amp, dur))
        if len(kept) == 6:
            break
    kept.sort()
    print(f'{sta3} {day3}: {len(bursts)} bursts > 6x median envelope in {night_h[0]:02d}-{night_h[1]:02d} local, '
          f'2-40 s long; showing {len(kept)}')
    if kept:
        from scipy.signal import spectrogram as sgram
        fig, axes = plt.subplots(2, len(kept), figsize=(3.6 * len(kept), 5.2),
                                 gridspec_kw={'height_ratios': [1, 1.3]}, squeeze=False)
        for k, (a, b, amp, dur) in enumerate(kept):
            c = (a + b) // 2
            i0, i1 = int(c - 30 * SR), int(c + 30 * SR)
            seg = data[i0:i1]
            t = np.arange(len(seg)) / SR
            hl = (hours[i0] % 24)
            axes[0, k].plot(t, seg, lw=0.4, color='#333')
            axes[0, k].set_title(f'{int(hl):02d}:{int((hl % 1) * 60):02d} local, {dur:.0f} s, {amp:.0f}x night median',
                                 fontsize=8)
            axes[0, k].set_xlim(0, 60)
            axes[0, k].set_xticklabels([])
            axes[0, k].tick_params(labelsize=7)
            f_, t_, Sx = sgram(seg, fs=SR, nperseg=256, noverlap=224)
            Sx = 10 * np.log10(Sx + 1e-12)
            axes[1, k].pcolormesh(t_, f_, Sx, shading='auto', cmap='magma',
                                  vmin=np.percentile(Sx, 5), vmax=np.percentile(Sx, 99.5))
            axes[1, k].set_ylim(0, 50)
            axes[1, k].set_xlim(0, 60)
            axes[1, k].set_xlabel('s', fontsize=8)
            axes[1, k].tick_params(labelsize=7)
        axes[0, 0].set_ylabel('counts (raw)', fontsize=8)
        axes[1, 0].set_ylabel('Hz', fontsize=8)
        fig.suptitle(f'{sta3} {day3}: isolated transients during the quiet hours '
                     f'({night_h[0]:02d}-{night_h[1]:02d} local), single vehicle passes on Minnesota Dr', fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f'am_vehicle_passes_{sta3}_{day3}.png', dpi=130)
        plt.close(fig)

    for day in days:
        print(f'{day}: ' + '; '.join(f'M{e["mag"]:.1f} {e["dist_km"]:.0f} km at '
                                   f'{e["time"].datetime.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz):%H:%M} local'
                                   for e in events.get(day, [])[:5]))
    print(f'figures in {out}/')


if __name__ == '__main__':
    main()
