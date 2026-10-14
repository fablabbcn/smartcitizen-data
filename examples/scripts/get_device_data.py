#!/usr/bin/env python
'''
Load the data of a device from the Smart Citizen API, and optionally export it.

Usage:
    python get_device_data.py 18154 --days 2
    python get_device_data.py 18061 --blueprint sc_air --days 7 --frequency 10Min --export ~/Desktop
'''
import argparse
import asyncio

import pandas as pd

import scdata as sc
from scdata.tools.custom_logger import set_logger_level


async def main(args):
    # Devices with hardware in the platform get their blueprint from it; the others need one
    blueprint = {'blueprint': args.blueprint} if args.blueprint else {}
    device = sc.Device(params=sc.APIParams(id=args.device), **blueprint)
    device.options.min_date = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
    device.options.frequency = args.frequency

    print(f'Device {device.id}: {device.handler.json.name}, blueprint {device.blueprint}')
    if not await device.load():
        print('No data')
        return

    print(device.data.describe().T)
    if args.process:
        device.process()
    if args.export:
        # One file per export of the blueprint (<id>_data_<export>.csv)
        device.export(args.export, forced_overwrite=True)
        print(f'Exported to {args.export}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('device', type=int, help='Device id')
    parser.add_argument('--days', type=int, default=1, help='Days of data to load (default 1)')
    parser.add_argument('--frequency', default='1Min', help='Frequency of the readings (default 1Min)')
    parser.add_argument('--blueprint', default=None, help='Blueprint, for devices without hardware (e.g. sc_air)')
    parser.add_argument('--process', action='store_true', help='Calculate the channels of the blueprint')
    parser.add_argument('--export', default=None, help='Folder to export the data to')
    parser.add_argument('--log-level', default='WARNING', help='DEBUG, INFO, WARNING or ERROR')
    args = parser.parse_args()

    set_logger_level(args.log_level.upper())
    asyncio.run(main(args))
