#!/usr/bin/env python
'''
Long processing: process months of the backups of a device with a long blueprint (baselines).

Reads the window of the blueprint (meta.window_days, 90 days) from the backups, from the first day of
that month, processes it with the long blueprint (e.g. sc_air_baseline) filled with the sensors of the
device's hardware, runs its health checks and exports the result as parquet. This is what Smart Citizen
Flows runs every week for the hardware with a long blueprint.

Needs pip install "scdata[storage]" and the backups: S3 (S3_DATA_BUCKET and the AWS credentials) or a
local folder with the same layout (--root, devices/<id>/data/).

Usage:
    python long_processing.py 18154
    python long_processing.py 18154 --root ~/backups --days 30 --set NO2.lam=1e8
'''
import argparse
import json
import os

import pandas as pd
import requests

import scdata as sc
from scdata._config import config
from scdata.tools.custom_logger import set_logger_level


def parameter(text):
    ''' CHANNEL.kwarg=value, the value as json (numbers, booleans) or as text '''
    target, _, value = text.partition('=')
    channel, _, kwarg = target.partition('.')
    try:
        value = json.loads(value)
    except ValueError:
        pass
    return channel, kwarg, value


def main(args):
    blueprint = requests.get(f'{config._base_postprocessing_url}blueprints/{args.blueprint}.json', timeout=30).json()
    if blueprint.get('meta', {}).get('kind') != 'long':
        print(f'{args.blueprint} is not a long blueprint')
        return

    device = sc.Device(params=sc.APIParams(id=args.device))
    if not device.use_blueprint(args.blueprint, blueprint):
        print(f'Device {device.id} has no hardware to fill the blueprint with')
        return

    # Tune the parameters of some channels
    for channel_name, kwarg, value in args.set:
        channel = next((item for item in device.channels if item.name == channel_name), None)
        if channel is None:
            print(f'{channel_name} is not a channel of {args.blueprint}: {[item.name for item in device.channels]}')
            return
        channel.kwargs = dict(channel.kwargs or {}, **{kwarg: value})
        print(f'{channel_name}: {kwarg} = {value}')

    days = args.days or blueprint['meta'].get('window_days', 90)
    start = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).replace(day=1).strftime('%Y-%m-%d')
    print(f'Device {device.id}, {args.blueprint}: backups since {start}, sensors {device.required_sensors}')
    root = os.path.expanduser(args.root) if args.root else None
    if not device.load_from_storage(min_date=start, channels=device.required_sensors, root=root) or device.data.empty:
        print('No backups for the window')
        return
    print(f'{len(device.data)} rows')

    device.process()
    device.health_checks()
    print('Health checks:', [(check['name'], check['status']) for check in device.health['checks']])

    channels = [channel.name for channel in device.channels if channel.name in device.data.columns]
    print(device.data[channels].describe().T)
    path = os.path.join(args.output, f'{device.id}_{args.blueprint}.parquet')
    os.makedirs(args.output, exist_ok=True)
    device.data[channels].to_parquet(path)
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('device', type=int, help='Device id')
    parser.add_argument('--blueprint', default='sc_air_baseline', help='Long blueprint (default sc_air_baseline)')
    parser.add_argument('--days', type=int, default=None, help='Window in days (default: window_days of the blueprint)')
    parser.add_argument('--root', default=None, help='Local folder with the backups, instead of S3')
    parser.add_argument('--set', type=parameter, action='append', default=[], metavar='CHANNEL.KWARG=VALUE',
                        help='Change a parameter of a channel, e.g. NO2.lam=1e8 (can be repeated)')
    parser.add_argument('--output', default='.', help='Folder for the result (default: here)')
    parser.add_argument('--log-level', default='ERROR', help='DEBUG, INFO, WARNING or ERROR')
    args = parser.parse_args()

    set_logger_level(args.log_level.upper())
    main(args)
