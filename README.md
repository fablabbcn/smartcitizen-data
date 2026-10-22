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

### Tests

A test groups devices. Its name starts with the year and month it is created (`2026_10_MY_TEST`). Creating, storing and opening a test are separate steps:

```
test = sc.Test(name='MY_TEST', devices=[...])   # in memory
test.save()                                     # in processed/<year>/<month>/2026_10_MY_TEST
test.save(root='tests')                         # in ./tests/2026_10_MY_TEST instead
test.save(root='ask')                           # asks: scdata cache, current folder or another folder
test.save(overwrite=True)                       # replaces a stored test with the same name

test = sc.Test.open('MY_TEST')                  # by name, with or without the date, or by folder
test = sc.Test.open_or_create('MY_TEST', devices=[...], root='tests')
sc.tests('MY')                                  # stored tests whose name contains MY
```

Tests stored out of the cache are registered in `tests.json` in the data folder, so they are opened by name from anywhere. `sc.Test.open` fails if several tests match the name (for instance `2026_09_MY_TEST` and `2026_10_MY_TEST`): open them by their full name.

`test.json`, in the test folder, keeps the options of the test and, for each device, its params, source, blueprint (and `override_url_blueprint`), options (`min_date`, `max_date`, `frequency`...), channels and checks. After changing any of them, `test.save()` stores them again. The data loaded from the API is cached in the `cached` subfolder (`test.options.cache`). Loading the test again reads the cache first, and only requests the dates of each device that the cache doesn't have: none if it reaches `max_date`, the new data if there is no `max_date`, and the period before it if `min_date` was moved earlier. A test that is not saved is loaded without cache.

A test folder can be moved or copied: open it by its path (`sc.Test.open('path/to/2026_10_MY_TEST')`), which updates its path and registers it. `sc.tests(root=...)` and `sc.Test.open(name, root=...)` also look into a folder with tests that are not registered.

#### Upgrading from 1.x

| 1.x | 2.0 |
| --- | --- |
| `sc.Test(name='X', devices=[...])`: created or opened, depending on what was stored | `sc.Test.open_or_create('X', devices=[...])` |
| `sc.Test(name='X', devices=[...], force_recreate=True)` | `sc.Test(name='X', devices=[...]).save(overwrite=True)` |
| `sc.Test(name='X')` to open a test, asking which one if several names contain `X` | `sc.Test.open('X')`: the exact name, with or without the date |
| `test.history()` | `sc.tests()` |

Tests saved with 1.x open as they are. Their devices get the default options, which 1.x did not store: set them and `test.save()`.

Also, `.env` files will be picked from `~/.cache/scdata`. An [env.example](env.example) is provided. If you want to upload data to [Zenodo](http://zenodo.org), you will need to fill set an environment variable called `ZENODO_TOKEN` in your environment. The `.env` file is not mandatory for `scdata` to work. It only serves for authentication if you want to `POST` or `GET` data without throttling. You can still do `GET` requests but they will be throttled if not authenticated/authorized by the platform. See [user roles](https://developer.smartcitizen.me/#authentication) for more info.

### Using with jupyter lab (optional)

It can also be used with `jupyter lab` or `jupyter`. For this [install juypterlab](https://github.com/jupyterlab/jupyterlab).

## Contribute

Issues and PR more than welcome!

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)
