# Blueprints

Documentation in: https://docs.smartcitizen.me/Guides/data/Handling%20calibration%20data/

Note: these blueprints define the sensors that will be loaded - ideally, the blueprint homogenises the data and then allows you to automatically pass it on to `metrics.

## About IDs

IDs that start with X in the blueprint, are not in the SC API but they are interesting for other APIs.

## Kinds

`meta.kind` says how flows uses a blueprint. Each hardware lists the blueprints it uses: one or two, at most one of each kind, and not `long` with `backup` (long processing always backs up).

- `process` (`sc_air`): processed every few hours on the latest data (up to 1000 readings) from the Smart Citizen API, and posted back. Health checks run on the same data. No baselines: they need months of data.
- `long` (`sc_air_baseline`): processed every `every_days` days on the last `window_days` days of the device's backup in S3, with baselines (CO2 with ALS, electrochemical sensors with `alphasense_als`). Results are stored in S3 and served by flows, not posted. The device is always backed up.
- `backup`: only the backup of the data in S3 (devices of researchers).

The baseline parameters (`lam`, `p`, background values) are starting values: they need tuning per sensor type.
