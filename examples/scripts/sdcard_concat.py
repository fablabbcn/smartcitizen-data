#!/usr/bin/env python
'''
Join the SD card files of devices and post them to the Smart Citizen API.

Each folder holds the files of a device (YY-MM-DD.CSV and the .01, .02... files of restarts) and is
named with the device id. The joined data is saved as CONCAT.CSV in each folder, with the columns
renamed to the names scdata uses (by the sensor id of the header).

Without --post nothing is posted (dry run). Posting needs a token of the owner of the devices
(SC_BEARER in the environment, or in ~/.cache/scdata/.env).

Usage:
    python sdcard_concat.py ~/.cache/scdata/raw/13318 ~/.cache/scdata/raw/13319
    python sdcard_concat.py ~/.cache/scdata/raw/* --timezone Europe/Madrid --post
'''
import argparse
import asyncio
from os.path import basename, isdir, normpath

import scdata as sc
from scdata.io import sdcard_concat
from scdata.tools.custom_logger import set_logger_level


async def main(args):
    for folder in args.folders:
        name = basename(normpath(folder))
        if not isdir(folder) or not name.isdigit():
            print(f'Skipping {folder}: not a folder named with a device id')
            continue

        data = sdcard_concat(folder, output='CONCAT.CSV', timezone=args.timezone, rename=True)
        # None when a file cannot be read (see the log)
        if data is None or data.empty:
            print(f'{name}: no readings (or files that cannot be read)')
            continue
        print(f'{name}: {len(data)} readings, {data.index.min()} to {data.index.max()}')

        device = sc.Device(blueprint=args.blueprint, params=sc.APIParams(id=int(name)))
        device.data = data
        device.handler.data = device.data
        # With dry_run, only the payload of the first chunk is printed
        await device.post(columns='sensors', dry_run=not args.post)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+', help='Folders with the files of each device, named with its id')
    parser.add_argument('--timezone', default='UTC', help='Time zone of the files (default UTC)')
    parser.add_argument('--blueprint', default='sc_air', help='Blueprint of the devices (default sc_air)')
    parser.add_argument('--post', action='store_true', help='Post the data (default: dry run)')
    parser.add_argument('--log-level', default='WARNING', help='DEBUG, INFO, WARNING or ERROR')
    args = parser.parse_args()

    set_logger_level(args.log_level.upper())
    asyncio.run(main(args))
