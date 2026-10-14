Smart Citizen Data
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fablabbcn/smartcitizen-data-framework/master?filepath=%2Fexamples%2Fnotebooks)
[![PyPI version](https://badge.fury.io/py/scdata.svg)](https://badge.fury.io/py/scdata)
[![Python application](https://github.com/fablabbcn/smartcitizen-data/actions/workflows/python-multiple-versions.yml/badge.svg)](https://github.com/fablabbcn/smartcitizen-data/actions/workflows/python-multiple-versions.yml)

Welcome to **SmartCitizen Data**. This is a data analysis framework for working with sensor data in different ways:

- Interacting with several sensors APIs
- Clean data, export and calculate metrics
- Model sensor data and calibrate sensors
- Generate data visualisations - matplotlib, ~[plotly](https://plotly.com/)~ or [uplot](https://leeoniya.github.io/uPlot)
- Generate analysis reports in html or pdf and upload them to [Zenodo](http://zenodo.org)

## Installation

You can check it out in the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fablabbcn/smartcitizen-data-framework/master?filepath=%2Fexamples%2Fnotebooks) before installing if you want. Works with `Python 3.*` (tested between 3.9 and 3.11).

You can just run:

```
pip install scdata
```

Plots (`device.ts_plot()`, maps, panels...) need the plotting libraries, an optional extra. Without them scdata loads and processes data, and plotting raises an error that says how to install them:

```
pip install "scdata[plot]"
```

Parquet needs the storage extra: the backups (local or S3: `device.load_from_storage()`, `device.backup_to_storage()`) and exports in parquet (`device.export(..., file_format='parquet')`):

```
pip install "scdata[storage]"
```

Examples (scripts and notebooks) are in [examples](examples).

### Work on the source code

Simply clone the repository with:

```
git clone https://github.com/fablabbcn/smartcitizen-data.git
cd smartcitizen-data
```

Install `scdata` package with requirements:

```
python setup.py install
```

Or if you want to edit:

```
pip install --editable ".[dev]"
```

### Tokens and config

Blueprints, calibrations and sensor names are read from `BASE_POSTPROCESSING_URL`, the same variable as [smartcitizen-connector](https://github.com/fablabbcn/smartcitizen-connector). By default it is this repository on GitHub; set it to `https://flows.smartcitizen.me/api/v1/` to use the metadata served by Smart Citizen Flows.

Settings can be changed in `~/.config/scdata/config.yaml`. The file only keeps what differs from the defaults (the defaults of new versions apply after upgrading), for instance:

```
data:
  reload_metadata: false
paths:
  export: /home/me/exports
```

The data folder is `~/.cache/scdata` by default (`$XDG_CACHE_HOME/scdata` if it is set), with the `raw`, `processed`, `export`... folders inside. Each folder of `paths` is changed on its own, and only to a folder that exists: changing `data` does not move the others. To move them all, set `XDG_CACHE_HOME`.

Also, `.env` files will be picked from `~/.cache/scdata`. An [env.example](env.example) is provided. If you want to upload data to [Zenodo](http://zenodo.org), you will need to fill set an environment variable called `ZENODO_TOKEN` in your environment. The `.env` file is not mandatory for `scdata` to work. It only serves for authentication if you want to `POST` or `GET` data without throttling. You can still do `GET` requests but they will be throttled if not authenticated/authorized by the platform. See [user roles](https://developer.smartcitizen.me/#authentication) for more info.

### Using with jupyter lab (optional)

It can also be used with `jupyter lab` or `jupyter`. For this [install juypterlab](https://github.com/jupyterlab/jupyterlab).

## Contribute

Issues and PR more than welcome!

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)
