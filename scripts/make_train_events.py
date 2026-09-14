#!/usr/bin/env python3
"""
Turn the Alaska Railroad timetable (configs/arr_schedule.yaml) into an events
CSV for scripts/collect_continuous_windows.py --label-from events.

One row per (train, day, station) with the passage time in UTC and a
``station`` column; the collector uses only the rows for the station it is
processing, so one CSV serves every station along the line.

Usage:
  python scripts/make_train_events.py --days 2026-09-09 2026-09-12 \
      --out configs/events/arr_2026-09-09_2026-09-12.csv
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--schedule', default=str(REPO_ROOT / 'configs' / 'arr_schedule.yaml'))
    p.add_argument('--days', nargs='*', default=[], help='Local calendar days, YYYY-MM-DD')
    p.add_argument('--start', default=None, help='First day of a range (with --end), inclusive')
    p.add_argument('--end', default=None, help='Last day of a range, inclusive')
    p.add_argument('--stations', nargs='*', default=None,
                   help='Emit one row per station with the passage offset applied (station column); '
                        'default: all stations in the schedule offsets')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    with open(args.schedule, encoding='utf-8') as fh:
        sch = yaml.safe_load(fh)
    tz, utc = ZoneInfo(sch['tz']), ZoneInfo('UTC')
    days = list(args.days)
    if args.start and args.end:
        d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
        days += [str(d0 + timedelta(days=i)) for i in range((d1 - d0).days + 1)]
    if not days:
        p.error('give --days or --start/--end')
    rows = []
    for day in days:
        d = date.fromisoformat(day)
        for tr in sch['trains']:
            s0, s1 = (x if isinstance(x, date) else date.fromisoformat(str(x)) for x in tr['season'])
            if not (s0 <= d <= s1) or d.strftime('%a') not in tr['days']:
                continue
            hh, mm = map(int, tr['depot_time'].split(':'))
            local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)
            offsets = sch['station_offsets_min'].get(tr['direction'], {})
            stations = args.stations if args.stations is not None else list(offsets)
            for sta in stations:
                if sta not in offsets:
                    continue  # this train does not pass that station
                at = local + timedelta(minutes=offsets[sta])
                rows.append({'station': sta, 'time': at.astimezone(utc).strftime('%Y-%m-%dT%H:%M:%S'),
                             'duration_sec': 120, 'event_id': f'{tr["id"]}_{day}',
                             'depot_time_local': local.strftime('%H:%M'), 'offset_min': offsets[sta],
                             'direction': tr['direction'], 'name': tr['name']})
    ev = pd.DataFrame(rows).sort_values(['time', 'station'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(args.out, index=False)
    print(f'{len(ev)} station passages -> {args.out}')
    print(ev[['station', 'time', 'event_id', 'depot_time_local', 'offset_min']].to_string(index=False))


if __name__ == '__main__':
    main()
