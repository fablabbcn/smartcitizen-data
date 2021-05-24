from scdata.utils import std_out, dict_fmerge, clean
from scdata._config import config
from numpy import array
from pandas import DataFrame

def prepare(self, measurand, inputs, options = dict()):
    """
    Prepares a test for a regression model
    Parameters
    ----------
        measurand: dict
            measurand = {'8019043': ['NO2']}
        inputs: dict
            inputs per device and reading
                inputs = {'devicename': ['reading-1', 'reading-2']}
        options: dict
            Options including data processing. Defaults in config._model_def_opt
    Returns
    -------
        df = pandas Dataframe
        measurand_name = string
    """

    options = dict_fmerge(options, config._model_def_opt)

    # Measurand
    measurand_device = list(measurand.keys())[0]
    measurand_metric = measurand[measurand_device][0]
    measurand_name = measurand[measurand_device][0] + '_' + measurand_device

    df = DataFrame()
    df[measurand_name] = self.devices[measurand_device].readings[measurand_metric]

    for input_device in inputs.keys():
        combined_df = self.combine(devices = [input_device], readings = inputs[input_device])
        df = df.combine_first(combined_df)

    if options['common_avg']:

        common_channels = inputs[list(inputs.keys())[0]]
        for input_device in inputs.keys():
            common_channels = list(set(common_channels).intersection(set(inputs[input_device])))
        std_out(f'Performing avg in common columns {common_channels}')
        for channel in common_channels:
            columns_list = [channel + '_' + device for device in list(inputs.keys())]
            df[channel + '_AVG'] = df[columns_list].mean(axis = 1)

        df = df.loc[:, df.columns.str.contains("_AVG")| df.columns.str.contains(measurand_name)]

    if options['clean_na'] is not None:
        df = clean(df, options['clean_na'], how = 'any')

    return df, measurand_name

def normalise_vbls(df, refn):
    # Get labels and features as numpy arrays
    df = df.reindex(sorted(df.columns), axis = 1)
    labels = array(df[refn])
    features = array(df.drop(refn, axis = 1))

    return labels, features
