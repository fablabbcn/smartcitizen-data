from plotly.graph_objs import Heatmap, Layout, Figure
from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data, groupby_session
from plotly.offline import iplot

def heatmap_iplot(self, **kwargs):
    """
    Plots heatmap in plotly interactive plot
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
        Plotly figure
    """
    if config.framework == 'jupyterlab': renderers.default = config.framework

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
        formatting = config._heatmap_def_fmt['plotly']
    else:
        formatting = dict_fmerge(config._heatmap_def_fmt['plotly'], kwargs['formatting'])

    # Make it standard
    for trace in traces:
        if 'subplot' not in trace: traces[trace]['subplot'] = 1

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)
    n_subplots = len(subplots)

    gskwags = {'frequency_hours': formatting['frequency_hours']}    

    dfgb, labels, yaxis, channel = groupby_session(df, **gskwags)
    xticks = [i.strftime("%Y-%m-%d") for i in dfgb.resample(formatting['session']).mean().index]

     # Data
    data = [
        Heatmap(
            z=dfgb[channel],
            x=dfgb.index.date,
            y=dfgb['session'],
            # colorscale=colorscale
        )
    ]

    layout = Layout(
        title = formatting['title'],
        xaxis = dict(ticks=''),
        yaxis = dict(ticks='' , categoryarray=labels, autorange = 'reversed')
    )

    figure = Figure(data=data, layout=layout)

    if options['show']: iplot(figure)

    return figure