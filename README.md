iScape Sensor Analysis Framework
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)

Welcome to the [iScape Sensor Analysis Framework](https://docs.iscape.smartcitizen.me/Sensor%20Analysis%20Framework/). This repository is built with the purpose of helping on the analysis, calibration and general post-processing of the sensors on field tests and it aims to be the primary tool for manipulating the sensors' data.

![](https://i.imgur.com/CvUuWpL.gif)

You can find more information in the [Official iScape Documentation](https://docs.iscape.smartcitizen.me/Sensor%20Analysis%20Framework).

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)

## Compatibility

Python 2.7 and 3.6 compatible.
The original notebook (in master branch) was developed in Python 2.7. A Python 3 version of the notebook is currently being developed in the [python3 branch](https://github.com/fablabbcn/smartcitizen-iscape-data/tree/python3). This is meant to be used with [JupyterLab](https://github.com/jupyterlab/jupyterlab) for data analysis.

## Installation

Easiest installation is done through [Anaconda](https://docs.anaconda.com/anaconda/install/).

Simply clone the repository with:

```
git clone git@github.com:fablabbcn/smartcitizen-iscape-data.git
cd smartcitizen-iscape-data
```

Easiest requirements installation is done through the `environment.yml` file:

```
conda env create -f environment.yml
pip install --editable .
```

More information about managing environments in Anaconda can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html). The folder structure follows that of [CookieCutter for Data Science](https://drivendata.github.io/cookiecutter-data-science/).

## Run

To run the notebooks, simply type:

```
jupyter lab
```

This will open a web browser instance (by default [localhost:8888/lab]()) which gives access to the following set of tools:

- Data load and import from CSV, local tests and the [SmartCitizen API](https://api.smartcitizen.me/)
- Data handling and exploration with custom plots via [plotly](https://plot.ly/)
- Anomaly detection via [XGBoost regressors](https://xgboost.readthedocs.io/en/latest/)
- Sensor model development and exploration with ML tools:
    - Time series statistical study (seasonality, correlation, auto-correlation, etc)
    - OLS and ARIMA with [statsmodels](https://www.statsmodels.org/stable/index.html)
    - Random Forests and SVR for time series ([scikit-learn](http://scikit-learn.org/)) 
    - LSTMs and other Deep Learning time series model approaches ([keras](https://keras.io/) with [tensorflow](https://www.tensorflow.org/) backend)