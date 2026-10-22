# Examples

Scripts (`.py`) and notebooks (`.ipynb`) using scdata. Install it with the extras the examples need:

```
pip install scdata              # load, process and export data
pip install "scdata[plot]"      # plots (notebooks 03, 13, 17, 18, 19)
pip install "scdata[storage]"   # parquet: backups (local or S3) and exports (notebooks 01 and 21, get_s3_objects.py, long_processing.py)
```

Blueprints, calibrations and sensor names are read from `BASE_POSTPROCESSING_URL` (by default this repository on GitHub; `https://flows.smartcitizen.me/api/v1/` for Smart Citizen Flows). Posting data needs a token of the owner of the device in `SC_BEARER` (environment, or `~/.cache/scdata/.env`): the examples that post are dry runs unless told otherwise.

## Scripts

Run them with `--help` for their options.

- Load a device and export it: [scripts/get_device_data.py](scripts/get_device_data.py)
- Process a device and post the results (what Smart Citizen Flows runs every few hours): [scripts/process_device.py](scripts/process_device.py)
- Health checks of a group of devices: [scripts/run_health_metrics.py](scripts/run_health_metrics.py)
- Back up a device to S3 and read a period of the backups: [scripts/get_s3_objects.py](scripts/get_s3_objects.py)
- Long processing, baselines over months of backups: [scripts/long_processing.py](scripts/long_processing.py)
- Join SD card files and post them: [scripts/sdcard_concat.py](scripts/sdcard_concat.py)

## Notebooks

- Getting started: configuration, blueprints, load from the API and CSV, tests (create, save, open and find them), export: [notebooks/01_getting_started.ipynb](notebooks/01_getting_started.ipynb)
- Plots (matplotlib, uPlot and maps): [notebooks/03_data_visualisation.ipynb](notebooks/03_data_visualisation.ipynb)
- Processing: channels, your own functions, baselines: [notebooks/04_processing_data.ipynb](notebooks/04_processing_data.ipynb)
- Query the Smart Citizen API to find devices: [notebooks/08_querying_the_sc_api.ipynb](notebooks/08_querying_the_sc_api.ipynb)
- SD card files: join, rename and post: [notebooks/09_load_and_post.ipynb](notebooks/09_load_and_post.ipynb)
- Large format plots (PDF): [notebooks/13_pdf_largescale_plots.ipynb](notebooks/13_pdf_largescale_plots.ipynb)
- R basic example (R kernel, [smartcitizen-R-data](https://github.com/fablabbcn/smartcitizen-R-data)): [notebooks/15_R_basic.ipynb](notebooks/15_R_basic.ipynb)
- Process a device and post the results: [notebooks/16_device_processing.ipynb](notebooks/16_device_processing.ipynb)
- A project's devices (TwinAIR): channels, CO2 baseline, plots: [notebooks/17_twinair_demo.ipynb](notebooks/17_twinair_demo.ipynb)
- Interactive plots with uPlot: [notebooks/18_interactive_plots.ipynb](notebooks/18_interactive_plots.ipynb)
- Interactive panels (bokeh): [notebooks/19_interactive_plots_with_panel.ipynb](notebooks/19_interactive_plots_with_panel.ipynb)
- Health checks: summary, flags, clean data: [notebooks/20_health_checks.ipynb](notebooks/20_health_checks.ipynb)
- Long processing: backups, long blueprints, baselines: [notebooks/21_long_processing.ipynb](notebooks/21_long_processing.ipynb)
- Basic data exploration: [notebooks/learning/01_basic_data_exploration.ipynb](notebooks/learning/01_basic_data_exploration.ipynb)

The notebooks that use a test store it in the scdata cache (`~/.cache/scdata/processed/<year>/<month>`) and replace it on each run, except [notebooks/01_getting_started.ipynb](notebooks/01_getting_started.ipynb), which also stores one next to it, in `notebooks/tests` (ignored by git), and [notebooks/learning/01_basic_data_exploration.ipynb](notebooks/learning/01_basic_data_exploration.ipynb), which opens its test once created. List them with `sc.tests()`.

Notebooks in [notebooks/todo](notebooks/todo) are not up to date.
