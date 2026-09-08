"""
Catalog and waveform access for building labeled datasets by event type.

Everything that needs a network connection lives here so the rest of the
pipeline (src/data/windows.py, the trainer, the packager) stays testable
offline. Requires obspy.

Sources
-------
* Events: USGS ComCat via the FDSN event service. ComCat carries an
  ``eventtype`` field (earthquake, quarry blast, explosion, ice quake,
  landslide, avalanche, volcanic eruption, ...) that the Alaska Earthquake
  Center fills in routinely. That field is what makes non-earthquake
  classes trainable from verified ground truth.
* Stations and waveforms: IRIS for the AK broadband network, the
  Raspberry Shake FDSN server for the AM network.
* Onsets: ComCat P picks when the event has phase data, otherwise a
  TauP travel time, optionally refined with STA/LTA.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from obspy import UTCDateTime, read_events
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel

from .windows import preprocess_trace, cut_event_and_noise, sta_lta_onset

USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
RASPISHAKE_URL = "https://data.raspberryshake.org"

_TAUP = None


def _taup():
    global _TAUP
    if _TAUP is None:
        _TAUP = TauPyModel(model="iasp91")
    return _TAUP


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def usgs_client() -> Client:
    return Client("USGS")


def waveform_client(network: str) -> Client:
    """IRIS for professional networks, the Raspberry Shake server for AM."""
    if network.upper() == "AM":
        if "RASPISHAKE" in URL_MAPPINGS:
            return Client("RASPISHAKE")
        return Client(RASPISHAKE_URL)
    return Client("IRIS")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _region_bounds(center_lat: float, center_lon: float, radius_km: float) -> dict:
    lat_pad = radius_km / 111.0
    lon_pad = radius_km / (111.0 * float(np.cos(np.radians(center_lat))))
    return dict(minlatitude=round(center_lat - lat_pad, 4), maxlatitude=round(center_lat + lat_pad, 4),
                minlongitude=round(center_lon - lon_pad, 4), maxlongitude=round(center_lon + lon_pad, 4))


def count_events_by_type(eventtype: str, start: str, end: Optional[str] = None,
                         min_magnitude: float = 1.0, **bounds) -> int:
    """Count ComCat events of one type inside a lat/lon box (cheap query)."""
    params = dict(format="geojson", starttime=start, minmagnitude=min_magnitude,
                  eventtype=eventtype, **bounds)
    if end:
        params["endtime"] = end
    url = USGS_EVENT_URL.replace("/query", "/count") + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        body = r.read().decode().strip()
    if body.startswith("{"):
        import json
        return int(json.loads(body)["count"])
    return int(body)


def get_events_by_type(eventtypes: list[str], start: str, end: Optional[str],
                       min_magnitude: float, max_magnitude: float, bounds: dict,
                       max_events: Optional[int] = None):
    """Fetch ComCat events of the given types as an obspy Catalog.

    Tries obspy's client first (it validates ``eventtype`` against the USGS
    WADL); falls back to a raw FDSN request if the parameter is rejected.
    """
    end = end or UTCDateTime().strftime("%Y-%m-%d")
    eventtype = ",".join(eventtypes)
    kwargs = dict(starttime=UTCDateTime(start), endtime=UTCDateTime(end),
                  minmagnitude=min_magnitude, maxmagnitude=max_magnitude,
                  orderby="time", **bounds)
    try:
        cat = usgs_client().get_events(eventtype=eventtype, **kwargs)
    except Exception:
        params = dict(format="quakeml", starttime=start, endtime=end,
                      minmagnitude=min_magnitude, maxmagnitude=max_magnitude,
                      eventtype=eventtype, orderby="time", **bounds)
        url = USGS_EVENT_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as r:
            cat = read_events(io.BytesIO(r.read()), format="QUAKEML")
    if max_events is not None and len(cat) > max_events:
        cat = cat[:max_events]
    return cat


def event_summary(event) -> dict:
    o = event.preferred_origin() or event.origins[0]
    m = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
    rid = str(event.resource_id)
    event_id = rid.split("eventid=")[-1] if "eventid=" in rid else rid.rsplit("/", 1)[-1]
    return dict(event_id=event_id, time=o.time, latitude=o.latitude, longitude=o.longitude,
                depth_km=(o.depth or 0.0) / 1000.0, magnitude=(m.mag if m else np.nan),
                event_type=str(event.event_type or "unknown"))


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

def get_stations(network: str, start: str, bounds: dict,
                 channel_priority=("BHZ", "HHZ", "EHZ", "SHZ")) -> pd.DataFrame:
    """One vertical channel per station in the box, best channel first."""
    client = waveform_client(network)
    chans = ",".join(sorted({c[:2] + "?" for c in channel_priority}))
    inv = client.get_stations(network=network, starttime=UTCDateTime(start),
                              channel=chans, level="channel", **bounds)
    rows = []
    for net in inv:
        for sta in net:
            codes = {ch.code for ch in sta.channels}
            locs = {ch.code: ch.location_code for ch in sta.channels}
            pick = next((c for c in channel_priority if c in codes), None)
            if pick is None:
                continue
            rows.append(dict(network=net.code, station=sta.code, latitude=sta.latitude,
                             longitude=sta.longitude, channel=pick,
                             location=locs.get(pick, "") or ""))
    return pd.DataFrame(rows)


def nearby_stations(stations: pd.DataFrame, lat: float, lon: float, max_km: float) -> pd.DataFrame:
    if stations.empty:
        return stations
    d = [gps2dist_azimuth(lat, lon, r.latitude, r.longitude)[0] / 1000.0 for r in stations.itertuples()]
    out = stations.copy()
    out["distance_km"] = d
    return out[out["distance_km"] <= max_km].sort_values("distance_km")


# ---------------------------------------------------------------------------
# Onsets
# ---------------------------------------------------------------------------

_PICK_CACHE: dict[str, dict[str, UTCDateTime]] = {}


def comcat_p_picks(event_id: str) -> dict[str, UTCDateTime]:
    """All P picks for an event keyed by station code (one request per event)."""
    if event_id in _PICK_CACHE:
        return _PICK_CACHE[event_id]
    picks_by_station: dict[str, UTCDateTime] = {}
    try:
        cat = usgs_client().get_events(eventid=event_id, includearrivals=True)
    except Exception:
        cat = None
    if cat:
        ev = cat[0]
        origin = ev.preferred_origin() or ev.origins[0]
        picks = {str(p.resource_id): p for p in ev.picks}
        for arr in origin.arrivals:
            p = picks.get(str(arr.pick_id))
            if p is None or not p.waveform_id:
                continue
            phase = (arr.phase or p.phase_hint or "").upper()
            sta = p.waveform_id.station_code
            if phase.startswith("P") and sta not in picks_by_station:
                picks_by_station[sta] = p.time
        for p in ev.picks:  # picks without arrivals (some AEC events)
            sta = p.waveform_id.station_code if p.waveform_id else None
            if sta and sta not in picks_by_station and (p.phase_hint or "").upper().startswith("P"):
                picks_by_station[sta] = p.time
    _PICK_CACHE[event_id] = picks_by_station
    return picks_by_station


def comcat_p_pick(event_id: str, station: str) -> Optional[UTCDateTime]:
    """P pick for one station from the event's phase data, or None."""
    return comcat_p_picks(event_id).get(station)


def taup_p_time(origin_time: UTCDateTime, depth_km: float, distance_km: float) -> UTCDateTime:
    deg = kilometers2degrees(distance_km)
    arrivals = _taup().get_travel_times(source_depth_in_km=max(depth_km, 0.0),
                                        distance_in_degree=deg, phase_list=["p", "P", "Pn", "Pg"])
    if not arrivals:
        return origin_time + distance_km / 6.0
    return origin_time + min(a.time for a in arrivals)


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    sampling_rate: float = 100.0
    lowcut_hz: float = 2.0
    highcut_hz: float = 20.0
    event_pre_sec: float = 30.0
    event_post_sec: float = 90.0
    noise_pre_sec: float = 90.0
    noise_end_sec: float = 30.0
    train_len_sec: float = 60.0

    @classmethod
    def from_dict(cls, d: dict) -> "WindowConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def collect_windows(catalog, stations: pd.DataFrame, class_name: str, class_index: int,
                    noise_index: int, network: str, cfg: WindowConfig,
                    max_station_km: float, max_stations_per_event: int,
                    onset_mode: str = "auto", stalta_refine: bool = True,
                    noise_per_event: bool = True, log=print):
    """Download one vertical trace per (event, station) and cut windows.

    Returns (windows, labels, metadata_rows). Every event window gets one
    pre-onset noise window from the same trace (label ``noise_index``) so
    the noise class always contains the instrument/site mix of the event
    classes.

    onset_mode: 'pick' (ComCat P pick only, skip if none), 'taup' (travel
    time only), 'auto' (pick, else taup). STA/LTA refinement is applied
    to taup onsets, and to picks only when the refined onset is within 3 s
    of the pick.
    """
    client = waveform_client(network)
    fs = cfg.sampling_rate
    pre_pad = cfg.noise_pre_sec + 10.0
    post_pad = cfg.event_post_sec + 10.0

    windows, labels, rows = [], [], []
    for ei, ev in enumerate(catalog):
        info = event_summary(ev)
        near = nearby_stations(stations, info["latitude"], info["longitude"], max_station_km)
        if near.empty:
            continue
        got = 0
        for st in near.itertuples():
            if got >= max_stations_per_event:
                break
            onset, source = None, None
            if onset_mode in ("pick", "auto"):
                onset = comcat_p_pick(info["event_id"], st.station)
                source = "comcat_pick" if onset is not None else None
            if onset is None and onset_mode in ("taup", "auto"):
                onset = taup_p_time(info["time"], info["depth_km"], st.distance_km)
                source = "taup"
            if onset is None:
                continue

            t0 = onset - pre_pad
            t1 = onset + post_pad
            try:
                stream = client.get_waveforms(st.network, st.station, st.location or "*",
                                              st.channel, t0, t1)
            except Exception as exc:  # noqa: BLE001 - network errors are expected
                log(f"  {info['event_id']} {st.station}: no data ({type(exc).__name__})")
                continue
            if not stream:
                continue
            stream.merge(method=1, fill_value="interpolate")
            tr = stream[0]
            if tr.stats.npts < (pre_pad + post_pad) * tr.stats.sampling_rate * 0.95:
                continue
            data = preprocess_trace(tr.data, tr.stats.sampling_rate, fs,
                                    cfg.lowcut_hz, cfg.highcut_hz)
            onset_sample = int(round((onset - tr.stats.starttime) * fs))

            if stalta_refine:
                refined = sta_lta_onset(data, fs, onset_sample, search_sec=10.0)
                if refined is not None:
                    if source == "taup" or abs(refined - onset_sample) <= 3 * fs:
                        onset_sample = refined
                        source = source + "+stalta"

            event_w, noise_w = cut_event_and_noise(
                data, fs, onset_sample, cfg.event_pre_sec, cfg.event_post_sec,
                cfg.noise_pre_sec, cfg.noise_end_sec)
            if event_w is None:
                continue

            base = dict(event_id=info["event_id"], event_type=info["event_type"],
                        event_time=str(info["time"]), magnitude=info["magnitude"],
                        depth_km=info["depth_km"], network=st.network, station=st.station,
                        channel=st.channel, distance_km=float(st.distance_km),
                        onset_source=source, original_sampling_rate=float(tr.stats.sampling_rate))
            windows.append(event_w)
            labels.append(class_index)
            rows.append(dict(base, label_name=class_name, window_type="event",
                             onset_sample=int(round(cfg.event_pre_sec * fs)),
                             window_len=len(event_w)))
            if noise_per_event and noise_w is not None:
                windows.append(noise_w)
                labels.append(noise_index)
                rows.append(dict(base, label_name="Noise", window_type="noise",
                                 onset_sample=-1, window_len=len(noise_w)))
            got += 1
        log(f"[{ei + 1}/{len(catalog)}] {info['event_type']} M{info['magnitude']:.1f} "
            f"{info['event_id']}: {got} station windows")
    return windows, labels, rows


def window_config_dict(cfg: WindowConfig) -> dict:
    return asdict(cfg)
