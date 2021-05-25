from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import gridspec
from matplotlib import style
from seaborn import set_palette

from math import sqrt
from pandas import to_datetime
from sklearn.metrics import mean_squared_error

def ts_scatter(self, **kwargs):
    """
    Plots timeseries and scatter comparison in matplotlib
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
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Name of auxiliary electrode found in dataframe
    Returns
    -------
        Matplotlib figure containing timeseries and scatter plot with correlation
        coefficients on it
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
        formatting = config._ts_scatter_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._ts_scatter_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)

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

    # If empty, nothing to do here
    if df is None:
        return None

    n_subplots = len(subplots)

    fig = plt.figure(figsize = (formatting['width'],
                                formatting['height']))

    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, -1])

    feature_trace = subplots[0][0]
    ref_trace = subplots[0][1]

    # Calculate basic metrics
    pearsonCorr = list(df.corr('pearson')[list(df.columns)[0]])[-1]
    rmse = sqrt(mean_squared_error(df[feature_trace].fillna(0), df[ref_trace].fillna(0)))

    std_out (f'Pearson correlation coefficient: {pearsonCorr}')
    std_out (f'Coefficient of determination R²: {pearsonCorr*pearsonCorr}')
    std_out (f'RMSE: {rmse}')

    # Time Series plot
    ax1.plot(df.index, df[feature_trace], color = 'g', label = feature_trace, linewidth = 1, alpha = 0.9)
    ax1.plot(df.index, df[ref_trace], color = 'k', label = ref_trace, linewidth = 1, alpha = 0.7)
    ax1.axis('tight')

    # Correlation plot
    ax2.plot(df[ref_trace], df[feature_trace], 'go', label = feature_trace, linewidth = 1,  alpha = 0.3)
    ax2.plot(df[ref_trace], df[ref_trace], 'k', label =  '1:1 Line', linewidth = 0.2, alpha = 0.6)
    ax2.axis('tight')

    if formatting['title'] is not None:
        ax1.set_title('Time Series Plot for {}'.format(formatting['title']),
                                                        fontsize=formatting['title_fontsize'])
        ax2.set_title('Scatter Plot for {}'.format(formatting['title']),
                                                        fontsize=formatting['title_fontsize'])

    if formatting['grid'] is not None:
        ax1.grid(formatting['grid'])
        ax2.grid(formatting['grid'])

    if formatting['legend']:
        ax1.legend(loc="best")
        ax2.legend(loc="best")

    if formatting['ylabel'] is not None:
        ax1.set_ylabel(formatting['ylabel'])
        ax2.set_xlabel(formatting['ylabel'])
        ax2.set_ylabel(formatting['ylabel'])

    if formatting['xlabel'] is not None:
        ax1.set_xlabel(formatting['xlabel'])

    if formatting['yrange'] is not None:
        ax1.set_ylim(formatting['yrange'])
        ax2.set_xlim(formatting['yrange'])
        ax2.set_ylim(formatting['yrange'])

    if formatting['xrange'] is not None:
        ax1.set_xlim(to_datetime(formatting['xrange']))

    if options['show']: plt.show();

    return fig
