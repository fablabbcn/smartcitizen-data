#!/usr/bin/env python
'''
Process a device and post the results: the cycle Smart Citizen Flows runs every few hours.

Loads the data since the last processing, processes it with the blueprint of the device's hardware,
runs the health checks, posts the channels to the Smart Citizen API and saves the date of the last
processing. Without --post nothing is posted (dry run). Posting needs a token of the owner of the
device (SC_BEARER in the environment, or in ~/.cache/scdata/.env).

Usage:
    python process_device.py 18154
    python process_device.py 18154 --post
'''
import argparse
import asyncio
import json
import warnings

import pandas as pd

import scdata as sc
from scdata.tools.custom_logger import set_logger_level


async def main(args):
    device = sc.Device(params=sc.APIParams(id=args.device))
    if not device.valid_for_processing:
        print(f'Device {device.id} has no valid postprocessing (hardware) in the platform')
        return

    latest = device.handler.json.postprocessing.latest_postprocessing
    device.options.min_date = (pd.Timestamp(latest) if latest else pd.Timestamp.now(tz='UTC') - pd.Timedelta('1D')).isoformat()
    device.options.limit = args.limit
    print(f'Device {device.id} ({device.blueprint}): loading from {device.options.min_date}')

    if not await device.load():
        print('No new data')
        return
    print(f'Loaded {len(device.data)} readings, until {device.data.index[-1]}')

    if not device.process():
        print('Processing failed')
        return

    device.health_checks()
    print('Health checks:', json.dumps([(check['name'], check['status']) for check in device.health['checks']]))

    dry_run = not args.post
    await device.post(columns='channels', dry_run=dry_run)
    if device.update_postprocessing_date():
        device.update_postprocessing(dry_run=dry_run)
    print('Done' + (' (dry run: nothing posted)' if dry_run else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('device', type=int, help='Device id')
    parser.add_argument('--limit', type=int, default=500, help='Readings to load at most (default 500)')
    parser.add_argument('--post', action='store_true', help='Post the results (default: dry run)')
    parser.add_argument('--log-level', default='WARNING', help='DEBUG, INFO, WARNING or ERROR')
    args = parser.parse_args()

    set_logger_level(args.log_level.upper())
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    asyncio.run(main(args))
