from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data, colors

# uPlot
from IPython.display import HTML
import json
from jinja2 import Template
import io
import re

'''
This code is heavily inspired by https://github.com/saewoonam/uplot_lib
'''

def ts_uplot(self, **kwargs):
    """
    Plots timeseries in uplot interactive plot - Fast, fast fast
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
        uPlot figure
    """

    head_template = '''
        <link rel="stylesheet" href="https://leeoniya.github.io/uPlot/dist/uPlot.min.css">
        <script src="https://leeoniya.github.io/uPlot/dist/uPlot.iife.js"></script>

        <div style="text-align:center">
            <h2 style="font-family: Roboto"> {{title}} </h2>
        </div>

        '''

    uplot_template = '''
        <div id="plot{{subplot}}"></div>
        <script>
            data = {{data}};
            options = {{options}};

            if (typeof options.scatter == 'undefined') {
                options.scatter = false
            }

            if (options.scatter) {
                for (i=1; i<data.length; i++) {
                    options['series'][i]["paths"] = u => null;
                }
            }

            u = new uPlot(options, data, document.getElementById("plot{{subplot}}"))
        </script>
        '''

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
        formatting = config._ts_plot_def_fmt['uplot']
    else:
        formatting = dict_fmerge(config._ts_plot_def_fmt['uplot'], kwargs['formatting'])

    # Size sanity check
    if formatting['width'] < 100:
        std_out('Setting width to 800')
        formatting['width'] = 800
    if formatting['height'] < 100:
        std_out('Reducing height to 600')
        formatting['height'] = 600

    if 'html' not in options:
        options['html'] = False

    h = Template(head_template).render(title=formatting['title'])

    # Get dataframe
    df, subplots = prepare_data(self, traces, options)

    # If empty, nothing to do here
    if df is None:
        return None

    df = df.fillna('null')
    n_subplots = len(subplots)

    # Get data in uplot expected format
    udf = df.copy()
    udf.index = udf.index.astype(int)/10**9

    for isbplt in range(n_subplots):

        sdf = udf.loc[:, subplots[isbplt]]
        sdf = sdf.reset_index()
        data = sdf.values.T.tolist()

        labels = sdf.columns
        useries = [{'label': labels[0]}]

        if formatting['ylabel'] is None:
            ylabel = None
        else:
            ylabel = formatting['ylabel'][isbplt+1]

        uaxes = [
                    {
                        'label': formatting['xlabel'],
                        'labelSize': formatting['fontsize'],
                    },
                    {
                        'label': ylabel,
                        'labelSize': formatting['fontsize']
                    }
                ]

        color_idx=0

        for label in labels:
            if label == labels[0]: continue
            if color_idx+1>len(colors): color_idx=0

            nser = {
                    'label': label,
                    'stroke': colors[color_idx],
                    'points': {'space': 0, 'size': formatting['size']}
                    }

            useries.append(nser)
            color_idx += 1

        u_options = {
                        'width': formatting['width'],
                        'height': formatting['height'],
                        'legend': {'isolate': True},
                        'cursor': {
                                    'lock': True,
                                    'focus': {
                                                'prox': 16,
                                    },
                                    'sync': {
                                                'key': 'moo',
                                                'setSeries': True,
                                    },
                                    'drag': {
                                                'x': True,
                                                'y': True,
                                                'uni': 50,
                                                'dist': 10,
                                            }
                                    },
                        'scales': {
                                    'x': {'time': True},
                                    'y': {'auto': True},
                                  },
                        'series': useries,
                        'axes': uaxes
                    }

        h2 = Template(uplot_template).render(data=json.dumps(data),
                                 options=json.dumps(u_options),
                                 subplot=isbplt)


        h += h2

    h = h.replace('"', "'")
    h = h.replace("'null'", "null")

    if options['html']:
        return h
    else:
        iframe = f'''<iframe srcdoc="{h}" src=""
            frameborder="0" width={formatting['width'] + formatting['padding-right']}
            height={formatting['height'] + formatting['padding-bottom']}
            sandbox="allow-scripts">
            </iframe>'''

        return HTML(iframe)
