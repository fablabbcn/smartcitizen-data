from scdata.utils import std_out, dict_fmerge
from .scatter_plot import scatter_plot
from scdata._config import config
from plotly.io import renderers
import plotly.tools as tls

def scatter_iplot(self, **kwargs):
    """
    Plots Correlation in plotly plot. Calls corr_plot and then converts it
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
            Name of auxiliary electrode found in dataframe. Defaults in config._corr_plot_def_fmt
    Returns
    -------
        Plotly figure
    """
    std_out ('Not yet working', 'ERROR')
    return None
    if config._framework == 'jupyterlab': renderers.default = config._framework

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
        formatting = config._scatter_plot_def_fmt['plotly']
    else:
        formatting = dict_fmerge(config._scatter_plot_def_fmt['plotly'], kwargs['formatting'])

    # Set options to not show in scatter_plot
    toshow = options['show']
    options['show'] = False
    
    # Make sns plot
    mfig = scatter_plot(self, traces = traces, options = options, formatting = formatting)
    options['show'] = toshow
    
    pfig = tls.mpl_to_plotly(mfig);

    if options['show']: pfig.show();	        

    return pfig