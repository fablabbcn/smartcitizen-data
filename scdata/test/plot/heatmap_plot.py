import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from seaborn import set_palette, heatmap
from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data, groupby_session

def heatmap_plot(self, **kwargs):
    """
    Plots heatmap in seaborn plot, based on period binning
    Parameters
    ----------
        traces: dict
            Data for the plot, with the format:
            "traces":  {"1": {"devices": '8019043',
                             "channel" : "PM_10"}
                        }
        options: dict
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Name of auxiliary electrode found in dataframe. Defaults in config._heatmap_def_fmt
    Returns
    -------
        Matplotlib figure
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
        options = config._plot_def_opt
    else:
        options = dict_fmerge(config._plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config._heatmap_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._heatmap_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)

    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});

    # Make it standard
    for trace in traces:
        if 'subplot' not in trace: traces[trace]['subplot'] = 1

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)

    # If empty, nothing to do here
    if df is None:
        return None

    n_subplots = len(subplots)

    gskwags = {'frequency_hours': formatting['frequency_hours']}

    dfgb, labels, yaxis, _ = groupby_session(df, **gskwags)

    # Sample figsize in inches
    _, ax = plt.subplots(figsize=(formatting['width'], formatting['height']));


    xticks = [i.strftime("%Y-%m-%d") for i in dfgb.resample(formatting['session']).mean().index]

    # Pivot with 'session'
    g = heatmap(dfgb.pivot(columns='session').resample(formatting['session']).mean().T, ax = ax,
                cmap = formatting['cmap'], robust = formatting['robust'],
                vmin = formatting['vmin'], vmax = formatting['vmax'],
                xticklabels = xticks, yticklabels = labels);

    # ax.set_xticks(xticks*ax.get_xlim()[1]/(2))
    _ = g.set_xlabel(formatting['xlabel']);
    _ = g.set_ylabel(yaxis);

    # Set title
    _ = g.figure.suptitle(formatting['title'], fontsize=formatting['title_fontsize']);
    plt.subplots_adjust(top=formatting['suptitle_factor'])

    # Show
    if options['show']: plt.show()

    return g.figure
