SmartCitizen Sensor Data Framework
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)

Welcome to the **SmartCitizen Sensor Data Framework**. This is a framework built with the purpose of *analysis*, *calibration* and *post-processing* of sensors data, related to any field, but particularly focused on air-quality data coming from low-cost sensors. It aims to unify several sources of data and to provide tools for analysing data, creating reports and more.

![](assets/images/saf_schema.png)

## Features

A full documentation of the framework is detailed in [the Smart Citizen Docs](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/). Some features include:

- Interacting with several sensors APIs (see [here](scdata/data/api.py))
- Clean data, export and calculate metrics
- Model sensor data and calibrate sensors
- Generate data visualisations:

![](assets/images/covid-noise.png)

- Generate analysis reports and upload them to [Zenodo](http://zenodo.org)

![](assets/images/Workflow.png)

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)

## Compatibility

Works with `Python 3.7`. It can be used with [JupyterLab](https://github.com/jupyterlab/jupyterlab) for data analysis, but it is not mandatory. There are plenty of examples in the `examples` folder. These are meant to show how to interface with the `python` code in the `scdata` folder.

## Installation

Simply clone the repository with:

```
git clone git@github.com:fablabbcn/smartcitizen-iscape-data.git
cd smartcitizen-iscape-data
```

Install requirements with:

```
pip install -r requirements.txt
```

The code in the framework is managed as internal dependencies. To activate this, you can run:

```
pip install --editable . --verbose
```

### Tokens and config

If you want to upload data to [Zenodo](http://zenodo.org), you will need to fill the `secrets.py` with a `token` as below:

```
ZENODO_TOKEN='your-token'
```

You can get more instructions [here](https://docs.smartcitizen.me/Guides/Upload%20data%20to%20zenodo/).

### Windows

[xgboost](https://pypi.org/project/xgboost/) is not currently supported in pip for windows users, and needs to be [directly installed from github](https://xgboost.readthedocs.io/en/latest/build.html). Remove the line in the `environment.yml` to avoid installation issues. 

## Contribute

Issues and PR welcome!
