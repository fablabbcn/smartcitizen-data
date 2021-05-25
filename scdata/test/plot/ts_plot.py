from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data
from pandas import to_datetime

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from seaborn import set_palette
from numpy import where, array

def ts_plot(self, **kwargs):
    """
    Plots timeseries in matplotlib plot
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
            Formatting dict. Defaults in config._ts_plot_def_fmt
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
        formatting = config._ts_plot_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._ts_plot_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)

    # Palette
    if formatting['palette'] is not None: set_palette(formatting['palette'])

    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)

    # If empty, nothing to do here
    if df is None:
        return None

    n_subplots = len(subplots)

    # Size sanity check
    if formatting['width'] > 50:
        std_out('Reducing width to 12')
        formatting['width'] = 12
    if formatting['height'] > 50:
        std_out('Reducing height to 10')
        formatting['height'] = 10

    # Plot
    figure, axes = plt.subplots(n_subplots, 1,
                                sharex = formatting['sharex'],
                                figsize = (formatting['width'],
                                           formatting['height'])
                                );

    if n_subplots == 1:
        axes = array(axes)
        axes.shape = (1)

    for ax in axes:

        isbplt = where(axes == ax)[0][0];

        # Check if we are plotting any highlight for the trace
        if any(['-MEAN' in trace for trace in subplots[isbplt]]): has_hl = True
        elif any(['-MAX' in trace for trace in subplots[isbplt]]): has_hl = True
        elif any(['-MIN' in trace for trace in subplots[isbplt]]): has_hl = True
        else: has_hl = False

        for trace in subplots[isbplt]:

            if has_hl:
                if '-MEAN' in trace: alpha = formatting['alpha_highlight']
                elif '-MAX' in trace: alpha = formatting['alpha_highlight']
                elif '-MIN' in trace: alpha = formatting['alpha_highlight']
                else: alpha = formatting['alpha_other']
            else: alpha = 1

            ax.plot(df.index, df[trace], label = trace, alpha = alpha);

        # TODO make this to compare to not None, so that we can send location
        if formatting['legend']:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
        if formatting['ylabel'] is not None:
            ax.set_ylabel(formatting['ylabel'][isbplt+1]);
        if formatting['xlabel'] is not None:
            ax.set_xlabel(formatting['xlabel']);
        if formatting['yrange'] is not None:
            ax.set_ylim(formatting['yrange'][isbplt+1]);
        if formatting['xrange'] is not None:
            if formatting['sharex']: ax.set_xlim(to_datetime(formatting['xrange'][1]));
            else: ax.set_xlim(to_datetime(formatting['xrange'][isbplt+1]));

        if formatting['grid'] is not None:
            ax.grid(formatting['grid']);

        if formatting["decorators"] is not None:

            if 'axvline' in formatting['decorators']:
                for vline in formatting['decorators']['axvline']:
                    ax.axvline(to_datetime(vline), linestyle = 'dotted', color = 'gray');

            if 'axhline' in formatting['decorators']:
                for vline in formatting['decorators']['axhline']:
                    ax.axhline(vline, linestyle = 'dotted', color = 'gray');

            if 'xtext' in formatting['decorators']:
                for xtext in formatting['decorators']['xtext'].keys():
                    text = formatting['decorators']['xtext'][xtext]
                    position = formatting['yrange'][isbplt+1][1]-(formatting['yrange'][isbplt+1][1]-formatting['yrange'][isbplt+1][0])/10
                    ax.text(to_datetime(xtext), position, text, size=15, color = 'gray');

            # TODO Fix
            if 'ytext' in formatting['decorators']:
                for ytext in formatting['decorators']['ytext'].keys():
                    text = formatting['decorators']['ytext'][ytext]
                    position = formatting['xrange'][isbplt+1][1]-(formatting['xrange'][isbplt+1][1]-formatting['yrange'][isbplt+1][0])/10
                    ax.text(ytext, position, text, size=15, color = 'gray');

    figure.suptitle(formatting['title'], fontsize=formatting['title_fontsize']);
    plt.subplots_adjust(top = formatting['suptitle_factor']);

    if options['show']: plt.show();

    return figure
