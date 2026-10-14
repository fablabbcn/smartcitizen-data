#!/usr/bin/env python
'''
Health checks of a group of devices: gaps, flat values, implausible values and outliers, as the
blueprint of each device describes them.

For each device it writes to the output folder:
- <id>_health.json: the summary of device.health (per check and column, the readings or time flagged,
  the ratio and the first periods flagged)
- <id>_qc_data.csv.gz: the data with a flag column per check and column (with --qc-data)
and a summary.csv with the highest ratio flagged by each check per device.

Devices are given by id, or read from an Excel file (sheet DEVICES, column id) filtered by a column.

Usage:
    python run_health_metrics.py --devices 18154 17592 --days 7
    python run_health_metrics.py --excel deployments.xlsx --criterion Location --value Thriassio --days 30
'''
import argparse
import asyncio
import json
import logging
import os
import warnings
from os.path import join

import pandas as pd

import scdata as sc
from scdata.tools.custom_logger import set_logger_level

logger = logging.getLogger('health_metrics')


def devices_from_excel(path, criterion, value):
    devices = pd.read_excel(path, sheet_name='DEVICES')
    devices['id'] = pd.to_numeric(devices['id'], errors='coerce')
    devices = devices.dropna(subset=['id'])
    if criterion:
        devices = devices[devices[criterion] == value]
    return [int(device_id) for device_id in devices['id']]


async def check_device(device_id, args, semaphore):
    ''' Loads, processes and checks a device. Returns {check: highest ratio flagged}, or None '''
    async with semaphore:
        try:
            blueprint = {'blueprint': args.blueprint} if args.blueprint else {}
            device = sc.Device(params=sc.APIParams(id=device_id), **blueprint)
            device.options.min_date = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
            if not await device.load():
                logger.warning(f'{device_id}: no data')
                return None
        except Exception:
            logger.exception(f'{device_id}: cannot load')
            return None

    # Processing and checks are CPU work: in a thread, so that other devices keep loading
    def checks():
        device.process()
        device.health_checks()
        with open(join(args.output, f'{device_id}_health.json'), 'w') as file:
            json.dump(device.health, file, indent=2)
        if args.qc_data:
            device.qc_data.to_csv(join(args.output, f'{device_id}_qc_data.csv.gz'))
        return {check['name']: max([summary['ratio'] for summary in check['columns'].values()], default=0)
                for check in device.health['checks']}

    try:
        result = await asyncio.to_thread(checks)
    except Exception:
        logger.exception(f'{device_id}: checks failed')
        return None
    logger.info(f'{device_id}: {result}')
    return result


async def main(args):
    device_ids = args.devices or devices_from_excel(args.excel, args.criterion, args.value)
    logger.info(f'Checking {len(device_ids)} devices, last {args.days} days')
    os.makedirs(args.output, exist_ok=True)

    # A few devices at a time: the Smart Citizen API throttles many requests
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[check_device(device_id, args, semaphore) for device_id in device_ids])

    summary = pd.DataFrame({device_id: result for device_id, result in zip(device_ids, results) if result is not None}).T
    summary.index.name = 'device'
    summary.to_csv(join(args.output, 'summary.csv'))
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--devices', type=int, nargs='+', help='Device ids')
    source.add_argument('--excel', help='Excel file with a DEVICES sheet and an id column')
    parser.add_argument('--criterion', help='Column of the Excel file to filter by (e.g. Location)')
    parser.add_argument('--value', help='Value of that column (e.g. Thriassio)')
    parser.add_argument('--days', type=int, default=7, help='Days of data to check (default 7)')
    parser.add_argument('--blueprint', default=None, help='Blueprint, for devices without hardware (e.g. sc_air)')
    parser.add_argument('--output', default='health_metrics', help='Output folder (default health_metrics)')
    parser.add_argument('--qc-data', action='store_true', help='Also write the flags (qc_data) of each device')
    parser.add_argument('--concurrency', type=int, default=3, help='Devices loaded at the same time (default 3)')
    parser.add_argument('--log-level', default='INFO', help='DEBUG, INFO, WARNING or ERROR')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    # scdata warns for every column of the blueprint a device does not have
    set_logger_level('ERROR')
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    asyncio.run(main(args))
