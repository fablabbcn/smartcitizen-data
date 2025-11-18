import logging
import numpy as np
import pandas as pd
import argparse
import asyncio
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    import scdata as sc
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
        device = sc.Device(blueprint='sc_air', params=sc.APIParams(id=device_id))

        device.options.min_date = "2025-01-01"
        device.options.max_date = None

        logger.info("Getting data for device %s", device_id)
        asyncio.run(device.load())
        
        logger.info("get_nan_ratio for device %s", device_id)
        device.get_nan_ratio().to_csv(f"twinair_health_metrics/device_{device_id}_nan_ratios.csv.gz")
        logger.info("get_implausible_ratio for device %s", device_id)
        device.get_implausible_ratio().to_csv(f"twinair_health_metrics/device_{device_id}_implausible_ratios.csv.gz")
        logger.info("get_outlier_ratio for device %s", device_id)
        device.get_outlier_ratio().to_csv(f"twinair_health_metrics/device_{device_id}_outlier_ratios.csv.gz")
        logger.info("get_top_value_ratio for device %s", device_id)
        device.get_top_value_ratio().to_csv(f"twinair_health_metrics/device_{device_id}_top_value_ratios.csv.gz")