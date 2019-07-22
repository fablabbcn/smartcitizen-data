iScape Sensor Analysis Framework
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)

Welcome to the [SmartCitizen-iScape Sensor Analysis Framework](https://docs.iscape.smartcitizen.me/Sensor%20Analysis%20Framework/). This repository is built with the purpose of helping on the *analysis*, *calibration* and general *post-processing of sensors on field tests and it aims to be the primary tool for manipulating the sensors' data.

![](https://i.imgur.com/CvUuWpL.gif)

You can find more information in the:
- [Official iScape Documentation](https://docs.iscape.smartcitizen.me/Sensor%20Analysis%20Framework).
- [Smart Citizen Documentation](ttps://docs.smartcitizen.me/Sensor%20Analysis%20Framework)

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)

## Compatibility

`Python 3.6` compatible. It can be used with [JupyterLab](https://github.com/jupyterlab/jupyterlab) for data analysis, but it is not mandatory. The notebooks in the `notebooks` folder are meant as examples and interfaces to the `python` code in the `src` folder.

## Installation

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

Verify that the `src.egg-link` is performed properly within your environment. You can check this in the final lines of the previous command.

Additional commands to install jupyter lab extensions are given in the `.dotfile`:

```
pip install ipywidgets --upgrade
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.0
jupyter labextension install @jupyterlab/toc
conda install -c conda-forge jupyter_nbextensions_configurator
```

You can run it by:

```
chmod +x .dotfile
./.dotfile
```

### Note

More information about managing environments in Anaconda can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html). The folder structure follows that of [CookieCutter for Data Science](https://drivendata.github.io/cookiecutter-data-science/).

To run the notebooks you will need `Jupyter Lab`. You can install it following the installation instructions in [the official documentation](https://github.com/jupyterlab/jupyterlab#installation).

## Run

For launching the interface simply type:

```
jupyter lab
```

This will open a web browser instance (by default [localhost:8888/lab]()) which gives access to the tools in the framework. The `analysis.ipynb` is meant as an example, but any configuration is possible, even without the `jupyter lab` interface (of course).

## Features

A full documentation of the framework is detailed in [the Smart Citizen Docs](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/). The documentation is constantly being updated, as well as the framework.

### Data load and cleaning

- `test` creation for sensor co-location and archiving
- Data load and import from CSV, local `test` and the [SmartCitizen API](https://api.smartcitizen.me/)
- Data caching for API tests, to avoid requesting to the API the same data over and over
- Data cleaning with anomaly detection via [XGBoost regressors](https://xgboost.readthedocs.io/en/latest/)
- Visual calculator for channels

### Data visualisation

- Time series exploration with custom plots via [plotly](https://plot.ly/) or `matplotlib` and custom export options for reporting
- Correlation analysis plots

### Data model and processing

- Sensor model development and exploration with ML tools:
    - Time series statistical study (seasonality, correlation, auto-correlation, etc)
    - OLS models with [statsmodels](https://www.statsmodels.org/stable/index.html)
    - Random Forests and SVR for time series ([scikit-learn](http://scikit-learn.org/)), including `gridsearch`
    - LSTMs and other Deep Learning time series model approaches ([keras](https://keras.io/) with [tensorflow](https://www.tensorflow.org/) backend)
- Sensor model storage, and comparison with statistical methods and plots
- Data analysis (processing and plots) in batch via `json` descriptor file, included in `tasks/*.json`. Used to run data analysis tasks for several `tests` of files in batch, avoiding manual processing (see the instructions [here](https://github.com/fablabbcn/smartcitizen-iscape-data/tree/master/tasks))
