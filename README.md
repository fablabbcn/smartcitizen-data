iScape Sensor Analysis Framework
=======

[![DOI](https://zenodo.org/badge/97752018.svg)](https://zenodo.org/badge/latestdoi/97752018)

Welcome to the [SmartCitizen-iScape Sensor Analysis Framework](https://docs.iscape.smartcitizen.me/Sensor%20Analysis%20Framework/). This repository is built with the purpose of helping on the *analysis*, *calibration* and general *post-processing of sensors on field tests and it aims to be the primary tool for manipulating the sensors' data.

![](https://i.imgur.com/CvUuWpL.gif)

You can find more information in the:
- [Official iScape Documentation](https://docs.iscape.smartcitizen.me/Sensor%20Analysis%20Framework)
- [Smart Citizen Documentation](ttps://docs.smartcitizen.me/Sensor%20Analysis%20Framework)

## Funding

This work has received funding from the European Union's Horizon 2020 research and innovation program under the grant agreement [No. 689954](https://cordis.europa.eu/project/rcn/202639_en.html)

## Compatibility

`Python 3.7` compatible. It can be used with [JupyterLab](https://github.com/jupyterlab/jupyterlab) for data analysis, but it is not mandatory. The notebooks in the `notebooks` folder are meant as examples and interfaces to the `python` code in the `src` folder.

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

Additional commands to install jupyter lab extensions are given in the `.dotfile`:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.0
jupyter labextension install @jupyterlab/toc
jupyter labextension install jupyterlab-plotly@1.0.0
conda install -c conda-forge jupyter_nbextensions_configurator
```

With an optional one for plotly chart studio:

```
jupyter labextension install jupyterlab-chart-editor@1.2
```

You can run it by:

```
chmod +x .dotfile
./.dotfile
```

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

## Features

A full documentation of the framework is detailed in [the Smart Citizen Docs](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/). The documentation is constantly being updated, as well as the framework.

## Contribute

PR welcome! Do not hesitate to work with 

Also, if you are using git with the notebook, you can use [nbdime](https://nbdime.readthedocs.io/en/latest/#) for an easier handling.