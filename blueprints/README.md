# Blueprints

Read more in [the official docs](https://docs.smartcitizen.me/guides/data/handling-calibration-data/).

## Kinds

`meta.kind` defines how [flows](https://github.com/fablabbcn/smartcitizen-flows/) uses a blueprint. Each hardware lists the blueprints it uses: one or two, at most one of each kind. `long` can't be run with `backup` (since `long` processing always backs up).

- `process` (`sc_air`): processed every few hours on the latest data (up to 1000 readings) from the Smart Citizen API, and posted back. Health checks run on the same data. This blueprint can't be used to process baselines, since they need months of data.
- `long` (`sc_air_baseline`): processed every `every_days` days on the last `window_days` days of the device's backup in S3, with baselines (CO2 with ALS (algorithmic least squares), electrochemical sensors with `alphasense_als`). The results are stored in S3 and served by flows, but they aren't posted back the SC API. The device is always backed up.
- `backup`: only the backup of the data in S3 (devices owned by researchers).

The baseline parameters (`lam`, `p`, background values) are starting values: they need tuning per sensor type.
