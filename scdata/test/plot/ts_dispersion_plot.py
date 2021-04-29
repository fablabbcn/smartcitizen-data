from scdata.utils import std_out
from scdata._config import config
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import rcParams
from matplotlib import style
from seaborn import set_palette
from scipy.stats import t
import numpy as np

def ts_dispersion_plot(self, **kwargs):
    '''
    Plots disperison timeseries in matplotlib plot
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
        formatting = config._ts_plot_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._ts_plot_def_fmt['mpl'], kwargs['formatting'])        

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
        std_out(f"Using limit for sigma confidence: {config._dispersion['limit_confidence_sigma']}")
    else:
        distribution = 't-student'
        std_out(f'Using t-student distribution.')

    # Size sanity check
    if formatting['width'] > 50: 
        std_out('Reducing width to 12')
        formatting['width'] = 12
    if formatting['height'] > 50:
        std_out('Reducing height to 10')
        formatting['height'] = 10        

    # Make subplot
    figure, (ax_tbr, ax_ok) = plt.subplots(nrows = 2, 
                                sharex = formatting['sharex'],
                                figsize = (formatting['width'],
                                           formatting['height'])
                                );
    # cmap = plt.cm.Reds
    norm = matplotlib.colors.Normalize(vmin=0, 
            vmax=config._dispersion['limit_errors']/2)
    ch_index = self.common_channels.index(channel)+1
    
    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)
    # Font size
    if formatting['fontsize'] is not None:
            rcParams.update({'font.size': formatting['fontsize']});

    total_number = len(self.common_channels)
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
        limit_confidence = t.interval(config._dispersion['t_confidence_level']/100.0, len(self.devices), 
                                        loc=self.dispersion_df[channel + '_AVG'], scale=dispersion_avg)
        upper_bound = limit_confidence[1]
        lower_bound = limit_confidence[0]

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

            if number_errors/max_number_errors > config._dispersion['limit_errors']/100:
                std_out (f"Device {device} out of {config._dispersion['limit_errors']}% limit\
                         - {np.round(number_errors/max_number_errors*100, 1)}% out", 'WARNING')
                alpha = 1
                ax_tbr.plot(self.dispersion_df.index, 
                         self.dispersion_df[ncol], 
                         color = 'r',
                         label = device, alpha = alpha)
            else:
                alpha = 1
                color = 'g'
                ax_ok.plot(self.dispersion_df.index, 
                         self.dispersion_df[ncol], 
                         color = color, 
                         label = device, alpha = alpha)

    # Add upper and low bound bound to subplot 1
    ax_tbr.plot(self.dispersion_df.index, self.dispersion_df[channel + '_AVG'],
            'b', label = 'Average', alpha = 0.6)
    ax_tbr.plot(self.dispersion_df.index, upper_bound,
            'k', label = 'Upper-Bound', alpha = 0.6)
    ax_tbr.plot(self.dispersion_df.index, lower_bound,
            'k',label = 'Lower-Bound', alpha = 0.6)

    # Format the legend
    lgd1 = ax_tbr.legend(bbox_to_anchor=(1, 0.5), fancybox=True,
            loc='center left', ncol = 5)
    ax_tbr.grid(True)
    ax_tbr.set_ylabel(channel + ' TBR')
    ax_tbr.set_xlabel('Time')

    # Add upper and low bound bound to subplot 2
    ax_ok.plot(self.dispersion_df.index, self.dispersion_df[channel + '_AVG'],
            'b', label = 'Average', alpha = 0.6)
    ax_ok.plot(self.dispersion_df.index, upper_bound,
            'k', label = 'Upper-Bound', alpha = 0.6)
    ax_ok.plot(self.dispersion_df.index, lower_bound,
            'k',label = 'Lower-Bound', alpha = 0.6)

    # Format the legend
    ax_ok.legend(bbox_to_anchor=(1, 0.5),
            fancybox=True, loc='center left', ncol = 5)
    lgd2 = ax_ok.legend(bbox_to_anchor=(1, 0.5),
            fancybox=True, loc='center left', ncol = 5)
    ax_ok.grid(True)
    ax_ok.set_ylabel(channel + ' OK')
    ax_ok.set_xlabel('Time')

    figure.suptitle(f'({ch_index}/{total_number}) - {channel}',
                 fontsize=formatting['title_fontsize'])
    plt.subplots_adjust(top = formatting['suptitle_factor']);

    if options['show']: plt.show()

    return figure    