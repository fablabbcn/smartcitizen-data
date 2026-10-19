import matplotlib.pyplot as plt
from matplotlib import rcParams, style
from pandas import concat
from scipy.cluster import hierarchy as hc
from seaborn import set_palette
from smartcitizen_connector.tools import pandas_freq

from scdata._config import config
from scdata.tools.cleaning import clean
from scdata.tools.custom_logger import logger
from scdata.tools.dictmerge import dict_fmerge


def ts_dendrogram(self, **kwargs):
    """
    Plots dendrogram of devices and channels in matplotlib plot. Takes all the channels
    in channels that are in the test `devices`
    Parameters
    ----------
        devices: list
            All the devices of the test (a device: itself)
            Ids of the devices
        channels: list
            'all'
            If 'all', uses all channels in the devices
        metric: string
            'correlation' for normal R2 or custom metric by callable
        'method': string
            'single'
            Method for dendrogram
        'options': dict
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Name of auxiliary electrode found in dataframe. Defaults in config._dendrogram_def_fmt
    Returns
    -------
        Dendrogram matrix, shows plot
    """
    if 'metric' not in kwargs: metric = 'correlation'
    else: metric = kwargs['metric']

    if 'method' not in kwargs: method = 'single'
    else: method = kwargs['method']

    # A test, or a device
    by_id = {device.id: device for device in getattr(self, 'devices', [self])}
    if 'devices' not in kwargs: devices = list(by_id)
    else: devices = kwargs['devices']

    if 'channels' not in kwargs: channels = 'all'
    else: channels = kwargs['channels']

    if 'options' not in kwargs:
        logger.info('Using default options')
        options = config._plot_def_opt
    else:
        options = dict_fmerge(config._plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        logger.info('Using default formatting')
        formatting = config._dendrogram_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._dendrogram_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)

    # Palette
    if formatting['palette'] is not None: set_palette(formatting['palette'])

    # Size sanity check
    if formatting['width'] > 50:

        logger.info('Reducing width to 12')
        formatting['width'] = 12

    if formatting['height'] > 50:

        logger.info('Reducing height to 10')
        formatting['height'] = 10

    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});

    # One row per channel of each device, one column per time
    series = []
    for device in devices:
        dfd = by_id[device].data.select_dtypes('number')
        if options['frequency'] is not None:
            dfd = dfd.resample(pandas_freq(options['frequency'])).mean()

        for channel in (dfd.columns if channels == 'all' else channels):
            if channel in dfd.columns: series.append(dfd[channel].rename(f'{device}_{channel}'))
    df = concat(series, axis = 1).T

    if options['clean_na'] is not None:
        # Times without all the channels, or filled from the previous time
        if options['clean_na'] == 'drop': df = df.dropna(axis = 1)
        if options['clean_na'] == 'fill': df = df.ffill(axis = 1)

    Z = hc.linkage(df, method = method, metric = metric)

    # Plot dendogram
    plt.figure(figsize=(formatting['width'], formatting['height']))
    plt.title(formatting['title'], fontsize = formatting['title_fontsize'])
    plt.subplots_adjust(top = formatting['suptitle_factor']);
    plt.xlabel(formatting['xlabel'])
    plt.ylabel(formatting['ylabel'])
    hc.dendrogram(
        Z,
        orientation=formatting['orientation'],
        leaf_font_size=formatting['fontsize'],  # font size for the x axis labels
        labels=df.index
    )

    if options['show']: plt.show()

    return Z
