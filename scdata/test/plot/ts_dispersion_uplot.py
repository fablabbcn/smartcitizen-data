from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import colors
from scipy.stats import t
import numpy as np

# uPlot
from IPython.display import HTML
import json
from jinja2 import Template
import io
import re

def ts_dispersion_uplot(self, **kwargs):
    '''
    Plots dispersion timeseries in uplot plot
    Parameters
    ----------
        channel: string
            Channel
        options: dict
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Formatting dict. Defaults in config._ts_plot_def_fmt
    Returns
    -------
        Matplotlib figure
    '''

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

    if 'channel' not in kwargs:
        std_out('Needs at least one channel to plot')
        return None
    else:
        channel = kwargs['channel']

    if 'options' not in kwargs:
        std_out('Using default options')
        options = config._plot_def_opt
    else:
        options = dict_fmerge(config._plot_def_opt, kwargs['options'])

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config._ts_plot_def_fmt['uplot']
    else:
        formatting = dict_fmerge(config._ts_plot_def_fmt['uplot'],
                                 kwargs['formatting'])

    # Size sanity check
    if formatting['width'] < 100:
        std_out('Setting width to 800')
        formatting['width'] = 800
    if formatting['height'] < 100:
        std_out('Reducing height to 600')
        formatting['height'] = 600

    if 'html' not in options:
        options['html'] = False

    if self.dispersion_df is None:
        std_out('Perform dispersion analysis first!', 'ERROR')
        return None

    if self.common_channels == []: self.get_common_channels()

    if channel not in self.common_channels:
        std_out(f'Channel {channel} not in common_channels')
        return None
    if channel in config._dispersion['ignore_channels']:
        std_out(f'Channel {channel} ignored per config')
        return None

    if len(self.devices)>config._dispersion['nt_threshold']:
        distribution = 'normal'
        std_out('Using normal distribution')
        std_out(f"Using limit for sigma confidence:\
                {config._dispersion['limit_confidence_sigma']}")
    else:
        distribution = 't-student'
        std_out(f'Using t-student distribution.')

    ch_index = self.common_channels.index(channel)+1
    total_number = len(self.common_channels)
    h = Template(head_template).render(title = f'({ch_index}/{total_number}) - {channel}')

    dispersion_avg = self._dispersion_summary[channel]

    if distribution == 'normal':
        limit_confidence = config._dispersion['limit_confidence_sigma']
        # Calculate upper and lower bounds
        if (config._dispersion['instantatenous_dispersion']):
            # For sensors with high variability in the measurements, it's better to use this
            upper_bound = self.dispersion_df[channel + '_AVG']\
                        + limit_confidence * self.dispersion_df[channel + '_STD']
            lower_bound = self.dispersion_df[channel + '_AVG']\
                        - abs(limit_confidence * self.dispersion_df[channel + '_STD'])
        else:
            upper_bound = self.dispersion_df[channel + '_AVG']\
                        + limit_confidence * dispersion_avg
            lower_bound = self.dispersion_df[channel + '_AVG']\
                        - abs(limit_confidence * dispersion_avg)
    else:
        limit_confidence = t.interval(config._dispersion['t_confidence_level']/100.0,
                                      len(self.devices),
                                      loc=self.dispersion_df[channel + '_AVG'],
                                      scale=dispersion_avg)
        upper_bound = limit_confidence[1]
        lower_bound = limit_confidence[0]

    udf = self.dispersion_df.copy()
    udf['upper_bound'] = upper_bound
    udf['lower_bound'] = lower_bound

    udf = udf.fillna('null')
    # List containing subplots. First list for TBR, second for OK
    subplots = [[],[]]

    if formatting['join_sbplot']: n_subplots = 1
    else: n_subplots = 2
    udf.index = udf.index.astype(int)/10**9

    # Compose subplots lists
    for device in self.devices:
        ncol = channel + '-' + device

        if ncol in self.dispersion_df.columns:

            # Count how many times we go above the upper bound or below the lower one
            count_problems_up = self.dispersion_df[ncol] > upper_bound
            count_problems_down =  self.dispersion_df[ncol] < lower_bound

            # Count them
            count_problems = [1 if (count_problems_up[i] or count_problems_down[i])\
                                else 0 for i in range(len(count_problems_up))]

            # Add the trace in either
            number_errors = np.sum(count_problems)
            max_number_errors = len(count_problems)

            # TBR
            if number_errors/max_number_errors > config._dispersion['limit_errors']/100:
                std_out (f"Device {device} out of {config._dispersion['limit_errors']}% limit\
                         - {np.round(number_errors/max_number_errors*100, 1)}% out", 'WARNING')
                subplots[0].append(ncol)
            #OK
            else:
                subplots[n_subplots-1].append(ncol)

    # Add upper and low bound bound to subplot 0
    subplots[0].append(channel + '_AVG')
    subplots[0].append('upper_bound')
    subplots[0].append('lower_bound')

    if n_subplots > 1:
        # Add upper and low bound bound to subplot 1
        subplots[n_subplots-1].append(channel + '_AVG')
        subplots[n_subplots-1].append('upper_bound')
        subplots[n_subplots-1].append('lower_bound')

        ylabels = [channel + '_TBR', channel + '_OK']
    else:
        ylabels = [channel]

    # Make subplots
    for isbplt in range(n_subplots):

        sdf = udf.loc[:, subplots[isbplt]]
        sdf = sdf.reset_index()
        data = sdf.values.T.tolist()

        labels = sdf.columns
        useries = [{'label': labels[0]}]

        ylabel = ylabels[isbplt]

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
            # Gray bounds and averages
            if '_bound' in label or '_AVG' in label:
                stroke = 'gray'
                point = {'space': 50, 'size': min([formatting['size'] - 2, 1])}
            else:
                stroke = colors[color_idx]
                point = {'space': 0, 'size': formatting['size']}


            nser = {
                    'label': label,
                    'stroke': stroke,
                    'points': point
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


