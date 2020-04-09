SmartCitizen Sensor Data Framework
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)

Welcome to the **SmartCitizen Sensor Data Framework**. This is a framework built with the purpose of *analysis*, *calibration* and *post-processing* of sensors data, related to any field, but particularly focused on air-quality data coming from low-cost sensors. It aims to be the primary tool for manipulating sensors data.

![](assets/images/saf_schema.png)

## Features

A full documentation of the framework is detailed in [the Smart Citizen Docs](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/). The documentation is constantly being updated, as well as the framework. Mainly, the framework is used for:
- Interacting with several sensors APIs (see [here](src/data/api.py))
- Clean data, export and calculate metrics
- Model sensor data and calibrate sensors
- Generate analysis reports and upload them to [Zenodo](http://zenodo.org)
- Generate data visualisations:

![](assets/images/covid-noise.png)

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)

## Compatibility

Works with `Python 3.7`. It can be used with [JupyterLab](https://github.com/jupyterlab/jupyterlab) for data analysis, but it is not mandatory. There are plenty of examples in the `examples` folder. These are meant to show how to interface with the `python` code in the `src` folder.

## Installation

Note: additionally, the [official docs](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/guides/Install%20the%20framework/) have extra information on installation.

Easiest installation is done through [Anaconda](https://docs.anaconda.com/anaconda/install/). The installation of Anaconda for different platforms is widely supported and documented. An `environment.yml` file is provided.

Simply clone the repository with:

```
git clone git@github.com:fablabbcn/smartcitizen-iscape-data.git
cd smartcitizen-iscape-data
```

And then install requirements installation is done through the `environment.yml` file:

```
conda env create -f environment.yml
```

The code in the framework is managed as internal dependencies. To activate this, you can run:

```
pip install --editable . --verbose
```

**Note:**

Verify that the `src.egg-link is performed properly within your environment. You can check this in the final lines of the previous command.

Additional commands to install jupyter lab extensions are given in the `.dotfile`. You can run it by:

```
chmod +x .dotfile
./.dotfile
```

### Tokens and config

If you want to upload data to [Zenodo](http://zenodo.org), you will need to fill the `secrets.py` with a `token` as below:

```
ZENODO_TOKEN='your-token'
```

You can get more instructions [here](https://docs.smartcitizen.me/Guides/Upload%20data%20to%20zenodo/).

### Windows

[xgboost](https://pypi.org/project/xgboost/) is not currently supported in pip for windows users, and needs to be [directly installed from github](https://xgboost.readthedocs.io/en/latest/build.html). Remove the line in the `environment.yml` to avoid installation issues. 

### More info

More information about managing environments in Anaconda can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html). The folder structure follows that of [CookieCutter for Data Science](https://drivendata.github.io/cookiecutter-data-science/).

To run the notebooks you will need `Jupyter Lab`, although this is not a must. If you rather use the shell, you can launch scripts from there. If you want to use the interface that jupyter provides, you can install it following the installation instructions in [the official documentation](https://github.com/jupyterlab/jupyterlab#installation).

## Run

For launching the interface simply type:

```
jupyter lab
```

This will open a web browser instance (by default [localhost:8888/lab]()) which gives access to the tools in the framework. The `main.ipynb` is meant as an example, but any configuration is possible, even without the `jupyter lab` interface (of course).

## Contribute

Issues and PR welcome!
