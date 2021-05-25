from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data

# Plotly
import plotly.tools as tls
from plotly.subplots import make_subplots
# import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.io import renderers

def ts_iplot(self, **kwargs):
    """
    Plots timeseries in plotly interactive plot
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
            Name of auxiliary electrode found in dataframe. Defaults in config._ts_plot_def_fmt
    Returns
    -------
        Plotly figure
    """

    if config.framework == 'jupyterlab': renderers.default = config.framework

    if 'traces' not in kwargs:
        std_out('No traces defined', 'ERROR')
        return None
    else:
        traces = kwargs['traces']

    if 'options' not in kwargs:
        std_out('Using default options', 'WARNING')
        options = config._plot_def_opt
    else:
        options = dict_fmerge(config._plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting', 'WARNING')
        formatting = config._ts_plot_def_fmt['plotly']
    else:
        formatting = dict_fmerge(config._ts_plot_def_fmt['plotly'], kwargs['formatting'])

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)

    # If empty, nothing to do here
    if df is None:
        return None

    n_subplots = len(subplots)

    # Size sanity check
    if formatting['width'] < 100:
        std_out('Setting width to 800')
        formatting['width'] = 800
    if formatting['height'] < 100:
        std_out('Reducing height to 600')
        formatting['height'] = 600

    figure = make_subplots(rows = n_subplots, cols=1,
                           shared_xaxes = formatting['sharex'])

    # Add traces
    for isbplt in range(n_subplots):

        for trace in subplots[isbplt]:

            figure.append_trace({'x': df.index,
                                 'y': df[trace],
                                 'type': 'scatter',
                                 'mode': 'lines+markers',
                                 'name': trace},
                                isbplt + 1, 1)

        # Name the axis
        if formatting['ylabel'] is not None:
            figure['layout']['yaxis' + str(isbplt+1)]['title']['text'] = formatting['ylabel'][isbplt+1]

        if formatting['yrange'] is not None:
            figure['layout']['yaxis' + str(isbplt+1)]['range'] = formatting['yrange'][isbplt+1]

    # Add axis labels
    if formatting['xlabel'] is not None:
        figure['layout']['xaxis' + str(n_subplots)]['title']['text'] = formatting['xlabel']

    # Add layout
    figure['layout'].update(width = formatting['width'],
                            height = formatting['height'],
                            # legend = dict(
                            #             traceorder='normal',
                            #             font = dict(family='sans-serif',
                            #                         size=10,
                            #                         color='#000'),
                            #             xanchor = 'center',
                            #             orientation = 'h',
                            #             itemsizing = 'trace',
                            #             yanchor = 'bottom',
                            #             bgcolor ='rgba(0,0,0,0)',
                            #             bordercolor = 'rgba(0,0,0,0)',
                            #             borderwidth = 0),
                            title=dict(text=formatting['title'])
                           )

    if options['show']: figure.show()

    return figure
