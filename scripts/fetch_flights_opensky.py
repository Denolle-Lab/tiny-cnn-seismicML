#!/usr/bin/env python3
"""
Arrivals and departures at an airport from the OpenSky Network, written as an
events CSV for scripts/collect_continuous_windows.py --label-from events
(issue #11, aircraft class).

OpenSky's /flights/arrival and /flights/departure endpoints refuse anonymous
callers ("You cannot access historical flights"); a free account works for
the last 30 days, and researchers can ask OpenSky for unlimited history.
Credentials come from the environment: OPENSKY_USER / OPENSKY_PASSWORD
(classic accounts) or OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET (OAuth2
client credentials, newer accounts).

Times: OpenSky gives firstSeen (departure roll, roughly) and lastSeen
(touchdown, roughly), both to the second but with ADS-B coverage gaps, so
keep the collector's search-and-detect on (--event-search-sec 180 or so
around each time; a take-off roll or landing lasts 30 to 90 s).

Usage:
  python scripts/fetch_flights_opensky.py --airport PANC \
      --start 2026-09-09 --end 2026-09-13 --out configs/events/anc_flights_2026-09-09_2026-09-12.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API = 'https://opensky-network.org/api'
TOKEN_URL = 'https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token'


def session():
    s = requests.Session()
    s.headers['User-Agent'] = 'tiny-cnn-seismicML (Denolle-Lab)'
    cid, secret = os.environ.get('OPENSKY_CLIENT_ID'), os.environ.get('OPENSKY_CLIENT_SECRET')
    user, pw = os.environ.get('OPENSKY_USER'), os.environ.get('OPENSKY_PASSWORD')
    if cid and secret:
        r = requests.post(TOKEN_URL, data={'grant_type': 'client_credentials', 'client_id': cid,
                                           'client_secret': secret}, timeout=60)
        r.raise_for_status()
        s.headers['Authorization'] = f'Bearer {r.json()["access_token"]}'
    elif user and pw:
        s.auth = (user, pw)
    else:
        sys.exit('Set OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET or OPENSKY_USER/OPENSKY_PASSWORD '
                 '(free account at opensky-network.org); anonymous access has no flight history.')
    return s


def fetch(s, kind, airport, t0, t1):
    """One UTC day per request (the endpoint's limit); returns the raw flight dicts."""
    out = []
    t = t0
    while t < t1:
        te = min(t + timedelta(days=1), t1)
        for attempt in range(3):
            r = s.get(f'{API}/flights/{kind}', params={'airport': airport, 'begin': int(t.timestamp()),
                                                        'end': int(te.timestamp())}, timeout=120)
            if r.status_code == 200:
                out += r.json()
                break
            if r.status_code == 404:  # no flights in the interval
                break
            print(f'  {kind} {t:%Y-%m-%d}: HTTP {r.status_code} {r.text[:80]} (attempt {attempt + 1}/3)')
            time.sleep(10 * (attempt + 1))
        else:
            print(f'  {kind} {t:%Y-%m-%d}: giving up')
        t = te
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--airport', default='PANC', help='ICAO code (PANC = Ted Stevens Anchorage)')
    p.add_argument('--start', required=True, help='UTC date, inclusive')
    p.add_argument('--end', required=True, help='UTC date, exclusive')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    t0 = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    t1 = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    s = session()
    rows = []
    for kind, key in (('departure', 'firstSeen'), ('arrival', 'lastSeen')):
        flights = fetch(s, kind, args.airport, t0, t1)
        print(f'{kind}s: {len(flights)}')
        for f in flights:
            ts = datetime.fromtimestamp(f[key], tz=timezone.utc)
            rows.append({'time': ts.strftime('%Y-%m-%dT%H:%M:%S'), 'duration_sec': 90,
                         'event_id': f'{kind}_{(f.get("callsign") or f["icao24"]).strip()}_{ts:%Y%m%dT%H%M}',
                         'kind': kind, 'callsign': (f.get('callsign') or '').strip(), 'icao24': f['icao24'],
                         'other_airport': f.get('estArrivalAirport') if kind == 'departure' else f.get('estDepartureAirport')})
    ev = pd.DataFrame(rows).sort_values('time')
    ev.to_csv(args.out, index=False)
    print(f'{len(ev)} events -> {args.out}')


if __name__ == '__main__':
    main()
