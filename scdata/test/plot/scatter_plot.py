import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from seaborn import set_palette, jointplot
from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data

def scatter_plot(self, **kwargs):
    """
    Plots correlation in matplotlib plot
    Parameters
    ----------
        traces: dict
            Data for the plot, with the format:
            traces = {
                        "1": {"devices": "10751",
                              "channel": "EXT_PM_A_1"},
                        "2": {"devices": "10751",
                              "channel": "EXT_PM_A_10"
                              }    
                    } 
        options: dict 
            Options including data processing prior to plot. Defaults in config.plot_def_opt
        formatting: dict
            Name of auxiliary electrode found in dataframe. Defaults in config.scatter_plot_def_fmt
    Returns
    -------
        Matplotlib figure and axes
    """

    if config.framework == 'jupyterlab': plt.ioff();
    plt.clf();

    if 'traces' not in kwargs: 
        std_out('No traces defined', 'ERROR')
        return None
    else:
        traces = kwargs['traces']

    if 'options' not in kwargs:
        std_out('Using default options')
        options = config.plot_def_opt
    else:
        options = dict_fmerge(config.plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config.scatter_plot_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config.scatter_plot_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config.plot_style)
    
    # Palette
    if formatting['palette'] is not None: set_palette(formatting['palette'])
    
    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});

    # Make it standard
    for trace in traces:
        if 'subplot' not in trace: traces[trace]['subplot'] = 1

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)
    if len(subplots) > 1: std_out('Ignoring additional subplots. Make another plot', 'WARNING')

    g = jointplot(df[subplots[0][0]], df[subplots[0][1]], 
                    data=df, kind=formatting['kind'], 
                    height=formatting['height'])

    # Set title
    g.fig.suptitle(formatting['title'], fontsize = formatting['title_fontsize']);
    plt.subplots_adjust(top = formatting['suptitle_factor']);
    
    # if formatting['ylabel'] is not None: 
    #     g.ax_marg_x.set_ylabel(formatting['ylabel'][isbplt+1]);
    
    # if formatting['xlabel'] is not None: 
    #     g.ax_marg_x.set_xlabel(formatting['xlabel']);
    
    # Set y axis limit
    if formatting['yrange'] is not None: 
        g.ax_marg_y.set_xlim(formatting['xrange']);
        g.ax_marg_y.set_ylim(formatting['yrange']);
    
    # Set x axis limit
    if formatting['xrange'] is not None:
       g.ax_marg_x.set_xlim(formatting['xrange'])

    if options['show']: plt.show(); 

    return g.fig