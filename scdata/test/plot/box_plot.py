import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from seaborn import set_palette, boxplot
from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data, groupby_session

def box_plot(self, **kwargs):
    """
    Plots heatmap in seaborn plot, based on period binning
    Parameters
    ----------
        traces: dict
            Data for the plot, with the format:
            "traces":  {"1": {"devices": ['8019043', '8019044', '8019004'],
                             "channel" : "PM_10",
                             "subplot": 1,
                             "extras": ['max', 'min', 'avg']},
                        "2": {"devices": "all",
                             "channel" : "TEMP",
                             "subplot": 2}
                        }     
        options: dict 
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Name of auxiliary electrode found in dataframe. Defaults in config._boxplot_def_fmt
    Returns
    -------
        Matplotlib figure
    """

    if config._framework == 'jupyterlab': plt.ioff();
    plt.clf();

    if 'traces' not in kwargs: 
        std_out('No traces defined', 'ERROR')
        return None
    else:
        traces = kwargs['traces']

    if 'options' not in kwargs:
        std_out('Using default options')
        options = config._plot_def_opt
    else:
        options = dict_fmerge(config._plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config._boxplot_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._boxplot_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)

    # Make it standard
    for trace in traces:
        if 'subplot' not in trace: traces[trace]['subplot'] = 1  
    
    # Palette
    if formatting['palette'] is not None: set_palette(formatting['palette'])
    
    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)
    n_subplots = len(subplots)

    dfgb, labels, xaxis, channel = groupby_session(df, frequency_hours = formatting['frequency_hours'], 
                                                       periods = formatting['periods'])

    # Sample figsize in inches
    _, ax = plt.subplots(figsize=(formatting['width'], formatting['height']));         
    
    # Pivot with 'session'
    if formatting['periods'] is not None:
        g = boxplot(x=dfgb['session'], y=dfgb[channel], hue = dfgb['period'], \
            ax=ax, palette = formatting['cmap']);
    else:
        g = boxplot(x=dfgb['session'], y=dfgb[channel], ax=ax, palette = formatting['cmap']);

    # TODO make this to compare to not None, so that we can send location       
    if formatting['ylabel'] is not None: 
        _ = g.set_ylabel(formatting['ylabel']);
    
    if formatting['grid'] is not None: 
         _ = g.grid(formatting['grid']);
    
    if formatting['yrange'] is not None: 
        ax.set_ylim(formatting['yrange']);
    
    _ = g.set_xlabel(xaxis);
  
    # Set title
    if formatting['title'] is not None:
        _ = g.figure.suptitle(formatting['title'], fontsize=formatting['title_fontsize']);
    
    # Suptitle factor
    if formatting['suptitle_factor'] is not None:
        plt.subplots_adjust(top = formatting['suptitle_factor']);
    
    # Show
    if options['show']: plt.show()

    return g.figure