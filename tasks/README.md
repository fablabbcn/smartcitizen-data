# Task automation

This folder contains the `json` files that describe the batch processes to be done by the scripts in `src/models/batch.py`. An example of the usage of this functionality is shown in `notebooks/batch_analysis.ipynb`:

```
# Load the object
from src.models.batch import batch_analysis\n",
tasks_file = '../tasks/iScape.json'\n",
batch_session = batch_analysis(tasks_file, verbose = True)
# Run the analysis
batch_session.run()
```

## Tasks sequence

The tasks will be run in the following order:

- Load data (if needed)
- Sanity checks, verifying that the `json` file contains all the necessary data
- Pre-process data
- Model data
- Export

## Possibilities

These tasks are intended to automatise data analysis tasks as the following:

- Load, process and export data
- Generate models and apply them, extracting metrics and comparing if they extrapolate to different set of sensors in different conditions
- Make plots for different metrics in an automatic way, and export their renders

## Json task description

The `tasks_file` loaded is a `json` containing keys for each task to be run. Several tasks can be included and will be run consecutively:

```
{
    "TASK_1":{...},
    "TASK_2":{...},
    "TASK_3":{...}
}
```

Each of the `tasks` contains different fields, depending on what the process is. 

### Define data

In case no model needs to be calculated, the data can be specified directly in the task:

```
{
    "TASK_1":{"data":{"TEST_1": ["DEVICE_11","DEVICE_12"],
                    "TEST_2": ["DEVICE_21", "DEVICE_22", "DEVICE_23"],
                    ...
                    },
        },

}
```

If a model is to be calculated, the data is defined within the model key as seen below.

### Options

`options` can be defined for different cases:

```
"options": {"model_target": "ALPHASENSE",
            "export_data": "Only Processed",
            "rename_export_data": true
            "target_raster": "1Min",
            "clean_na": true,
            "clean_na_method": "fill",
            "use_cache": true,
            }
```

- `model_target`: if the model is to be stored under a specific category of models under the `models/` folder
- `export_data`: if the processed data (after pre-processing and modeling) has to be exported. It will be saved in the corresponding `test` folder, under `processed`. Options are:
    + `None`: don't export anything
    + `All`: all channels in the `pandas dataframe`
    + `Only Generic`: Export only channels that are under the `data/interim/sensorNamesExport.json`
    + `Only Processed`: Export only channels that are tagged as `processed` under the `data/interim/sensorNamesExport.json`
- `rename_export_data`: Rename the exported channels for better readability using the file `data/interim/sensorNamesExport.json`
- `target_raster`: frequency at which load the data (as defined in `pandas` [here](# https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
))
- `clean_na`: clean or not NaN
- `clean_na_method`: `drop` or `fill` with back-forward filling
- `use_cache`: whether or not to use file chaching for the analysis. This adds a `cached` folder in the corresponding `test` directory, which allows faster download from the `API`. It is implemented so that the only data to be downloaded is the one that is not cached

### Pre-process data

Different preprocessing options can be defined. The most common one, for the sensors in the _Smart Citizen Project_ are the analysis of the _electrochemical sensors_ but others can be used and implemented, for later call from `/src/models/batch.py`.

```
    "TASK_1": {
        "pre-processing": {
                "alphasense": {"baseline_method": "deltas",
                                "parameters": [30, 45, 5],
                                "methods": {"CO": ["classic", "na"],
                                            "NO2": ["baseline", "single_aux"],
                                            "O3": ["baseline", "single_aux"]
                                            },
                                "overlapHours": 0
                            }
            },
    }
```

**Note**: this section is currently project specific for electrochemical sensors defined [here](https://docs.smartcitizen.me/Components/Gas%20Pro%20Sensor%20Board/Electrochemical%20Sensors)

- `baseline_method`: baseline method used to calculate pollutant concentration, as defined in [the documentation](https://docs.smartcitizen.me/Components/Gas%20Pro%20Sensor%20Board/Electrochemical%20Sensors/#sensor-calibration). It can be `deltas` or `als`, using the methodology described in [here](https://doi.org/10.1016/j.atmosenv.2016.10.024) for `deltas` or [here](https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf) for the Asymmetric Least Squares Smoothing (ALS). 
- `parameters`: parameters for the baseline determination:
    + in case of `deltas` method, is a `list` with [`min_delta`, `max_delta`, `interval`]
    + in case of `als` method, is another `dict` containing `lambda` and `p` parameters
- `methods`: a dict containing which method to use for each pollutant calculation (see example above). options for the calculation are `classic` from Alphasense Ltd. recommendations, or `baseline`, using the `baseline_method` specified before. Each of them can use different baselines (`single_aux`, `single_temp` or `single_hum`).

### Model

In the `model` sub-task, currently three possibilities are implemented:

- Ordinary least squares (`OLS`) regression
- Random Forest (`RF`) or Support Vector Regressor `SVR`
- Long-Short Term Memory ('LSTM')

In the case of having a `model` task, the `data` defined above is ignored, and only the one under `model: {"data":{}}` is used.

An example is shown below:

```
"model": {
        "model_name": "Random_Forest_100",
        "model_type": "RF",
        "data": {"train": {"2019-03_EXT_UCD_URBAN_BACKGROUND_API": ["5262"]},
                "test": {"2019-06_EXT_HASSELT_SCHOOL": ["9530", "9694"],
                        "2019-03_EXT_UCD_URBAN_BACKGROUND_API": ["5261", "5565"]
                        },
                "reference_device": "CITY_COUNCIL",
                "features": {"REF": "NO2_REF",
                            "A":  "GB_2W",
                            "B": "GB_2A",
                            "C": "HUM"
                            }
                },
        "hyperparameters": {"ratio_train": 0.75,
                            "n_estimators": 100,
                            "shuffle_split": true
                            },
        "target": "ALPHASENSE",
        "options": {"session_active_model": false,
                    "export_model": false,
                    "export_model_file": false,
                    "extract_metrics": true,
                    "save_plots": false,
                    "show_plots": false
                    }
        },

```

- `model_name`: model name to be saved
- `model_type`: 'RF', 'SVR', 'LSTM' or 'OLS'
- `data`: dict containing the data to use for training, and features description. Under `train`, we define which of the tests and devices is to be used for the model definition, with the format `{"TEST": ["DEVICE"]}`. Under `test`, we define a series of `test` in which we'll evaluate the model extracted from the `train` dataset.
    + `reference_device`: `device` that contains the reference data
    + `features`: dict of `devices` tagged as `REF`, `A`, `B`, `C`... to define the features of the model, being `REF` the reference channel in the `reference_device`
- `hyperparameters`: dict containing different hyperparameters, depending on the type of model:
    + For all:
        *  `ratio_train`: generic, train-test split ratio
    + Random Forest:
        * `n_estimators`: only for `RF`. number of forests to use
        * `shuffle_split`: only for `RF`. whether or not use shuffle split
    + LSTM:
        * `n_lags`: number of lags to account in the LSTM input
        * `epochs`: number of epochs
        * `batch_size`: batch size                
        * `verbose`: verbose output during training
        * `loss`: loss function ('mse' or others)
        * `optimizer`: optimizer to use (`adam` or others)
        * `layers`: specific layer structure
- `options`: different options for the model calculated
    + `session_active_model`: keep the model active after the task is completed
    + `export_model`: export the model (parameters, hyperparameters, weights) to the `model/model_type` folder after the task is completed
    + `export_model_file`: export the model file (not recommended for `RF`) fo the same folder as above
    + `export_metrics`: export metrics for the model or not
    + `save_plots`: save model plots or not
    + `show_plots`: show model plots or not