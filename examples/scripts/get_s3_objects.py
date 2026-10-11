#!/usr/bin/env python
'''
Back up the data of a device to S3, and read a period of the backups.

backup_to_storage appends the data of the device to s3://<S3_DATA_BUCKET>/devices/<id>/data/ (parquet)
with its metadata in devices/<id>/metadata.json. load_from_storage reads only a period and some
columns: the filter is pushed down to the parquet files.

Needs pip install "scdata[storage]" and, for S3, the environment variables:

    S3_DATA_BUCKET=
    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    AWS_DEFAULT_REGION=

--root reads a local folder with the same layout instead of S3 (devices/<id>/data/).

Usage:
    python get_s3_objects.py 18154 --backup --days 1
    python get_s3_objects.py 18154 --since 2026-09-01 --channels TEMP HUM NOISE_A
    python get_s3_objects.py 18154 --since 2026-09-01 --root ~/backups
'''
import argparse
import asyncio
import json
import os

import pandas as pd

import scdata as sc
from scdata.tools.custom_logger import set_logger_level


def metadata(device_id, path='devices'):
    ''' The metadata.json of a backup '''
    import boto3
    s3 = boto3.resource('s3')
    body = s3.Object(os.environ['S3_DATA_BUCKET'], f'{path}/{device_id}/metadata.json').get()['Body']
    return json.loads(body.read().decode('utf-8'))


async def main(args):
    device = sc.Device(params=sc.APIParams(id=args.device))

    if args.backup:
        if 'S3_DATA_BUCKET' not in os.environ:
            print('Backups are written to S3: set S3_DATA_BUCKET and the AWS credentials')
            return
        device.options.min_date = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
        if not await device.load():
            print('No data to back up')
            return
        print(f'Backing up {len(device.data)} readings')
        device.backup_to_storage(path='devices')

    if args.root is None and 'S3_DATA_BUCKET' not in os.environ:
        print('Set S3_DATA_BUCKET (and the AWS credentials) or --root to read backups')
        return
    if not device.load_from_storage(min_date=args.since, max_date=args.until, channels=args.channels,
                                    root=os.path.expanduser(args.root) if args.root else None):
        print('Nothing read')
        return
    print(device.data.index.min(), '-', device.data.index.max(), f'{len(device.data)} rows')
    print(device.data.describe().T)

    if args.root is None:
        print('Metadata:', metadata(device.id).get('name'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('device', type=int, help='Device id')
    parser.add_argument('--backup', action='store_true', help='Back up the last days of data first')
    parser.add_argument('--days', type=int, default=1, help='Days to back up (default 1)')
    parser.add_argument('--since', default=None, help='First date to read')
    parser.add_argument('--until', default=None, help='Last date to read')
    parser.add_argument('--channels', nargs='+', default=None, help='Columns to read (all by default)')
    parser.add_argument('--root', default=None, help='Local folder with the backups, instead of S3')
    parser.add_argument('--log-level', default='WARNING', help='DEBUG, INFO, WARNING or ERROR')
    args = parser.parse_args()

    set_logger_level(args.log_level.upper())
    asyncio.run(main(args))
