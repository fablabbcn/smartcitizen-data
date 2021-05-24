Smart Citizen Data
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fablabbcn/smartcitizen-data-framework/master?filepath=%2Fexamples%2Fnotebooks)
[![PyPI version](https://badge.fury.io/py/scdata.svg)](https://badge.fury.io/py/scdata)

Welcome to **SmartCitizen Data**. This is a data analysis framework for working with sensor data in different ways:

- Interacting with several sensors APIs
- Clean data, export and calculate metrics
- Model sensor data and calibrate sensors
- Generate data visualisations - matplotlib, [plotly](https://plotly.com/) or [uplot](https://leeoniya.github.io/uPlot)
- Generate analysis reports in html or pdf and upload them to [Zenodo](http://zenodo.org)

A full documentation of the framework is detailed in [the Smart Citizen Docs](https://docs.smartcitizen.me/Data/Data%20Analysis/). 

## Installation

You can check it out in the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fablabbcn/smartcitizen-data-framework/master?filepath=%2Fexamples%2Fnotebooks) before installing if you want. Works with `Python 3.*`.

You can just run:

```
pip install scdata
```

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
cd scdata
pip install --editable .
```

### Tokens and config

If you want to upload data to [Zenodo](http://zenodo.org), you will need to fill set an environment variable called `ZENODO_TOKEN` in your environment. You can get more instructions [here](https://docs.smartcitizen.me/Guides/data/Upload%20data%20to%20zenodo/) and with [this example](https://github.com/fablabbcn/smartcitizen-data/blob/master/examples/notebooks/06_upload_to_zenodo.ipynb).

A configuration file is available at `~/.config/scdata/config.yaml`, which contains a set of configurable variables to allow or not the local storage of relevant data in the data folder, normally in `~/.cache/scdata`:

```
data:
  cached_data_margin: 2
  load_cached_api: true
  reload_metadata: true
  store_cached_api: true
paths:
  config: /Users/username/.config/scdata
  data: /Users/username/.cache/scdata
  export: /Users/username/.cache/scdata/export
  interim: /Users/username/.cache/scdata/interim
  inventory: ''
  models: /Users/username/.cache/scdata/models
  processed: /Users/username/.cache/scdata/processed
  raw: /Users/username/.cache/scdata/raw
  reports: /Users/username/.cache/scdata/reports
  uploads: /Users/username/.cache/scdata/uploads
zenodo_real_base_url: https://zenodo.org
zenodo_sandbox_base_url: http://sandbox.zenodo.org
```

Also, `.env` files will be picked from `~/.cache/scdata`.

### Using with jupyter lab (optional)

It can also be used with `jupyter lab` or `jupyter`. For this [install juypterlab](https://github.com/jupyterlab/jupyterlab).

## Contribute

Issues and PR more than welcome!

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)
