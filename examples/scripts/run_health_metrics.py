import logging
import numpy as np
import pandas as pd
import argparse
import asyncio
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
import scdata as sc


async def _async_process_device(device_id, folder="twinair_health_metrics", min_date=None, max_date=None):
    """
    Async worker that loads a device and computes health metrics.

    Parameters:
    - device_id: int
    - folder: str - output folder for CSVs
    - min_date, max_date: optional date strings to set device.options
    - logger: optional logger instance

    Returns True on success, False on error or when no data.
    """
    _logger = logging.getLogger(__name__)

    try:
        device = sc.Device(blueprint='sc_air', params=sc.APIParams(id=device_id))
        if min_date is not None:
            device.options.min_date = min_date
        if max_date is not None:
            device.options.max_date = max_date

        _logger.info("Getting data for device %s", device_id)
        await device.load()

    except requests.exceptions.HTTPError as e:
        _logger.error("Failed to instantiate or load device %s: %s", device_id, e)
        return False
    except Exception:
        _logger.exception("Unexpected error loading device %s", device_id)
        return False

    if device.data.empty:
        _logger.warning("No data for device %s, skipping health metrics", device_id)
        return False

    try:
        _logger.info("Exporting device %s raw data", device_id)
        device.export(folder, True)
        _logger.info("get_nan_ratio for device %s", device_id)
        device.get_nan_ratio().to_csv(f"{folder}/device_{device_id}_nan_ratios.csv.gz")
        _logger.info("get_implausible_ratio for device %s", device_id)
        device.get_implausible_ratio().to_csv(f"{folder}/device_{device_id}_implausible_ratios.csv.gz")
        _logger.info("get_outlier_ratio for device %s", device_id)
        device.get_outlier_ratio().to_csv(f"{folder}/device_{device_id}_outlier_ratios.csv.gz")
        _logger.info("get_top_value_ratio for device %s", device_id)
        device.get_top_value_ratio().to_csv(f"{folder}/device_{device_id}_top_value_ratios.csv.gz")
    except Exception:
        _logger.exception("Failed processing metrics for device %s", device_id)
        return False

    return True


def process_device(device_id, folder="twinair_health_metrics", min_date=None, max_date=None):
    """Synchronous wrapper that runs the async device worker to completion.

    Call this from synchronous code. For parallel execution you can call
    `_async_process_device` directly inside an asyncio event loop.
    """
    return asyncio.run(_async_process_device(device_id, folder=folder, min_date=min_date, max_date=max_date))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TwinAIR device health metric calculations.")
    parser.add_argument(
        "source",
        help="Path to the Excel file containing device info",
    )
    parser.add_argument(
        "criterion",
        help="Name of the Column to filter devices (e.g. 'Location')",
    )

    parser.add_argument(
        "value",
        help="Value used to filter devices (e.g. 'Thriassio')",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    # configure logging from CLI
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    criterion = args.criterion
    value = args.value

    logger.info("Running scdata version %s", sc.__version__)

    source = args.source
    logger.info("Loading sensor list from file %s", source)

    criterion = args.criterion
    logger.info("Filtering by %s == %s", criterion, value)


    devices = pd.read_excel(source, sheet_name="DEVICES")
    devices["id"] = pd.to_numeric(devices["id"], errors="coerce")
    devices = devices.dropna(subset="id")
    devices["id"] = devices["id"].astype(int)

    logger.info(f"Found {len(devices['id'])} devices")

    these = devices[devices[criterion] == value]

    logger.info(f"Filtered devices, remaining {len(these['id'])}")

    for device_id in tqdm(these["id"]):
        logger.info("Processing device %s", device_id)
        # call the sync wrapper which runs the async worker to completion
        success = process_device(device_id, folder="twinair_health_metrics", min_date="2025-01-01", max_date=None)
        if not success:
            logger.warning("Processing failed for device %s, continuing", device_id)
            continue