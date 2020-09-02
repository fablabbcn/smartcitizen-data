from scipy.cluster import hierarchy as hc
from pandas import DataFrame
from scdata.utils import std_out, dict_fmerge, clean
from scdata._config import config
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style

def ts_dendrogram(self, **kwargs):
    """
    Plots dendrogram of devices and channels in matplotlib plot. Takes all the channels 
    in channels that are in the test `devices`
    Parameters
    ----------
        devices: list or string
            'all'
            If 'all', uses all devices in the test
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
            Name of auxiliary electrode found in dataframe. Defaults in config._ts_plot_def_fmt
    Returns
    -------
        Dendrogram matrix, shows plot
    """    
    if 'metric' not in kwargs: metric = 'correlation'
    else: metric = kwargs['metric']
        
    if 'method' not in kwargs: method = 'single'
    else: method = kwargs['method']
    
    if 'devices' not in kwargs: devices = list(self.devices.keys())
    else: devices = kwargs['devices']
    
    if 'channels' not in kwargs: channels = 'all'
    else: channels = kwargs['channels']

    if 'options' not in kwargs:
        std_out('Using default options')
        options = config._plot_def_opt
    else:
        options = dict_fmerge(config._plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config._ts_plot_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._ts_plot_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)
    
    # Palette
    if formatting['palette'] is not None: set_palette(formatting['palette'])

    # Size sanity check
    if formatting['width'] > 50: 

        std_out('Reducing width to 12')
        formatting['width'] = 12
    
    if formatting['height'] > 50: 

        std_out('Reducing height to 10')
        formatting['height'] = 10    

    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});
        
    df = DataFrame()
    
    for device in devices:
        dfd = self.devices[device].readings.copy()
        dfd = dfd.resample(options['frequency']).mean()
        
        if channels != 'all': 
            for channel in channels: 
                if channel in dfd.columns: df = df.append(dfd[channel].rename(device+'_'+channel))
        else: df = df.append(dfd)

    df = clean(df, options['clean_na'], how = 'any')    
            
    # if options['clean_na'] is not None:
    #     if options['clean_na'] == 'drop': df.dropna(axis = 1, inplace=True)
    #     if options['clean_na'] == 'fill': df = df.fillna(method='ffill')
    
    # Do the clustering        
    Z = hac.linkage(df, method = method, metric = metric)

    # Plot dendogram
    plt.figure(figsize=(formatting['width'], formatting['height']))
    plt.title(formatting['title'], fontsize = formatting['titlefontsize'])
    plt.subplots_adjust(top = formatting['suptitle_factor']);
    plt.xlabel(formatting['xlabel'])
    plt.ylabel(formatting['ylabel'])
    hac.dendrogram(
        Z,
        orientation=formatting['orientation'],
        leaf_font_size=formatting['fontsize'],  # font size for the x axis labels
        labels=df.index
    )
    
    plt.show()
    
    return Z