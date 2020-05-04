from plotly.graph_objs import Heatmap, Layout, Figure
from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data, groupby_session

def heatmap_iplot(self, **kwargs):
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
            Options including data processing prior to plot. Defaults in config.plot_def_opt
        formatting: dict
            Name of auxiliary electrode found in dataframe. Defaults in config.heatmap_def_fmt
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
        options = config.plot_def_opt
    else:
        options = dict_fmerge(config.plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config.heatmap_def_fmt['plotly']
    else:
        formatting = dict_fmerge(config.heatmap_def_fmt['plotly'], kwargs['formatting'])

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)
    n_subplots = len(subplots)

    dfgb, labels, yaxis, _ = groupby_session(df, formatting['frequency_hours'])
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
        xaxis = dict(ticks=xticks),
        yaxis = dict(ticks=labels , categoryarray=labels, autorange = 'reversed')
    )

    figure = Figure(data=data, layout=layout)

    if options['show_plot']: iplot(self.figure)

    return figure