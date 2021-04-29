from scdata.utils import std_out
from scdata._config import config
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import ceil
from matplotlib import gridspec

plt.style.use('seaborn-white')

def scatter_dispersion_grid(self, **kwargs):
    '''
    Plots disperison timeseries in matplotlib plot
    Parameters
    ----------
        channels: list
            Channel
        options: dict 
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Formatting dict. Defaults in config._ts_plot_def_fmt
    Returns
    -------
        Matplotlib figure
    '''
    if self.common_channels == []: self.get_common_channels()

    if 'channels' not in kwargs:
        std_out('Using common channels')
        channels = self.common_channels
    else:
        channels = kwargs['channels']

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

    if len(self.devices)>30:
        distribution = 'normal'
        std_out('Using normal distribution')
        std_out(f"Using limit for sigma confidence: {config._dispersion['limit_confidence_sigma']}")
    else:
        distribution = 't-student'
        std_out(f'Using t-student distribution.')
    
    # Number of subplots
    number_of_subplots = len(channels) 
    if number_of_subplots % 2 == 0: cols = 2
    else: cols = 2
    rows = int(ceil(number_of_subplots / cols))

    # Create grid
    gs = gridspec.GridSpec(rows, cols, wspace=0.9, hspace=0.4)
    figure = plt.figure(figsize=(cols*10,rows*5))
    figure.tight_layout()

    n = 0

    for channel in channels:

        if channel not in self.common_channels:
            std_out(f'Channel {channel} not in common_channels')
            continue
        if channel in config._dispersion['ignore_channels']:
            std_out(f'Channel {channel} ignored per config')
            continue       
        
        ax = figure.add_subplot(gs[n])
        n += 1
        
        dispersion_avg = self._dispersion_summary[channel]
     
        if distribution:
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
            
        for device in list(self.devices):
            color = cm.viridis.colors[round(list(self.devices).index(device)\
                                      *len(cm.viridis.colors)/len(list(self.devices)))]

            plt.scatter(self.dispersion_df[channel + '_AVG'], 
                          self.dispersion_df[channel + '-' + device], 
                          label = device, alpha = 0.3, color = color)
     
        plt.plot([min(self.dispersion_df[channel + '_AVG']), max(self.dispersion_df[channel + '_AVG'])], 
                  [min(self.dispersion_df[channel + '_AVG']), max(self.dispersion_df[channel + '_AVG'])], 
                  'r', label = 'AVG', alpha = 0.9, linewidth = 1.5)

        plt.plot([min(self.dispersion_df[channel + '_AVG']), max(self.dispersion_df[channel + '_AVG'])],
                  [min(lower_bound), max(lower_bound)], 
                  'g', label = 'AVG ± σSTD', alpha = 0.8, linewidth = 1.5)
        
        plt.plot([min(self.dispersion_df[channel + '_AVG']), max(self.dispersion_df[channel + '_AVG'])],
                  [min(upper_bound), 
                   max(upper_bound)], 
                  'g', alpha = 0.8, linewidth = 1.5)
        
        plt.legend(bbox_to_anchor=(1, 0.4), fancybox=True, loc='center left', ncol = 2)
        plt.xlabel('Refererence (avg. of test)')
        plt.ylabel('Individual device (-)')
        plt.title(f"Dispersion analysis for {channel} sensor - STD = {round(self.dispersion_df[channel + '_STD'].mean(), 2)}")
        plt.grid()
    
    plt.subplots_adjust(top = formatting['suptitle_factor']);

    if options['show']: plt.show()

    return figure
