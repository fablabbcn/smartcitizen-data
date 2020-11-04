import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style
from seaborn import set_palette, regplot
from scdata.utils import std_out, dict_fmerge
from scdata._config import config
from .plot_tools import prepare_data, colors
from numpy import array
from math import floor, ceil

def scatter_plot(self, **kwargs):
    """
    Plots correlation in matplotlib plot
    Parameters
    ----------
        traces: dict
            Data for the plot, with the format:
            traces = {1: {'devices': ['10751', '10751'],
                          'channels': ['TEMP', 'GB_2A'],
                          'subplot': 1},
                      2: {'devices': ['10752', '10752'],
                          'channels': ['TEMP', 'GB_2A'],
                          'subplot': 1}
                      3: {'devices': ['10751', '10751'],
                          'channels': ['TEMP', 'GB_2W'],
                          'subplot': 2}
                      4: {'devices': ['10752', '10752'],
                          'channels': ['TEMP', 'GB_2W'],
                          'subplot': 2} 
                    }
        options: dict 
            Options including data processing prior to plot. Defaults in config._plot_def_opt
        formatting: dict
            Formatting dict. Defaults in config._scatter_plot_def_fmt
    Returns
    -------
        Matplotlib figure and axes
    """

    if config._framework == 'jupyterlab': plt.ioff();
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
        formatting = config._scatter_plot_def_fmt['mpl']
    else:
        formatting = dict_fmerge(config._scatter_plot_def_fmt['mpl'], kwargs['formatting'])

    # Style
    if formatting['style'] is not None: style.use(formatting['style'])
    else: style.use(config._plot_style)
    
    # Palette
    if formatting['palette'] is not None: set_palette(formatting['palette'])
    
    # Font size
    if formatting['fontsize'] is not None: rcParams.update({'font.size': formatting['fontsize']});

    # Make it standard
    ptraces = dict()

    for trace in traces:
        if 'subplot' not in traces[trace]: traces[trace]['subplot'] = 1
        if 'channels' not in traces[trace]: ptraces = traces; continue
        
        ptrace_1 = trace * 10 + 1
        ptrace_2 = trace * 10 + 2

        ptraces[ptrace_1] = {'devices': traces[trace]['devices'][0],
                             'channel': traces[trace]['channels'][0],
                             'subplot': traces[trace]['subplot']
                            }

        ptraces[ptrace_2] = {'devices': traces[trace]['devices'][1],
                             'channel': traces[trace]['channels'][1],
                             'subplot': traces[trace]['subplot']
                            }                            


    # Get dataframe
    df, subplots = prepare_data(self, ptraces, options)
    n_subplots = len(subplots)

    # Plot
    nrows = min(n_subplots, formatting['nrows'])
    ncols = ceil(n_subplots/nrows)
    
    figure, axes = plt.subplots(nrows, ncols, figsize = (formatting['width'],
                                                          formatting['height'])
                                );

    if n_subplots == 1: 
        axes = array(axes)
        axes.shape = (1)
    
    cind = 0
    y_axes = list()
    x_axes = list()

    for i in subplots:
        for j in range(int(len(i)/2)):
            cind += 1
            if cind > len(colors)-1: cind = 0

            if nrows > 1 and ncols > 1:
                row = floor(subplots.index(i)/ncols)
                col = subplots.index(i)-row*ncols                
                ax = axes[row][col]
            else:
                ax = axes[subplots.index(i)]

            kwargs = {
                        'data':df, 
                        'ax': ax,
                        'label': f'{i[2*j+1]} vs. {i[2*j]}'
                    }

            if formatting['palette'] is None: kwargs['color'] = colors[cind]

            regplot(df[i[2*j]], df[i[2*j+1]], **kwargs)

            if formatting['legend']: 
                ax.legend(loc='best')

            if formatting['ylabel'] is not None:
                try:
                    ax.set_ylabel(formatting['ylabel'][subplots.index(i)+1]);
                except:
                    std_out (f'y_label for subplot {subplots.index(i)} not set', 'WARNING')
                    ax.set_ylabel('')
                    pass
            else:
                ax.set_ylabel('')

            if formatting['xlabel'] is not None: 
                try:
                    ax.set_xlabel(formatting['xlabel'][subplots.index(i)+1]);
                except:
                    std_out (f'x_label for subplot {subplots.index(i)} not set', 'WARNING')
                    ax.set_xlabel('')
                    pass    
            else:
                ax.set_xlabel('')

            y_axes.append(ax.get_ylim())
            x_axes.append(ax.get_xlim())

    # Unify axes or set what was ordered
    for i in subplots:
        for j in range(int(len(i)/2)):


            if nrows > 1 and ncols > 1:
                row = floor(subplots.index(i)/ncols)
                col = subplots.index(i)-row*ncols                
                ax = axes[row][col]
            else:
                ax = axes[subplots.index(i)]
           
            # Set y axis limit
            if formatting['yrange'] is not None and not formatting['sharey']:
                try:
                    ax.set_ylim(formatting['yrange']);
                except:
                    std_out (f'yrange for subplot {subplots.index(i)} not set', 'WARNING')
                    pass                
            elif formatting['sharey']:
                ax.set_ylim(min([yl[0] for yl in y_axes]), max([yl[1] for yl in y_axes]))
            
            # Set x axis limit
            if formatting['xrange'] is not None and not formatting['sharex']:
                try:
                    ax.set_xlim(formatting['xrange']);
                except:
                    std_out (f'xrange for subplot {subplots.index(i)} not set', 'WARNING')
                    pass                     
            elif formatting['sharex']:
                ax.set_xlim(min([xl[0] for xl in x_axes]), max([xl[1] for xl in x_axes])) 

    # Set title
    figure.suptitle(formatting['title'], fontsize = formatting['title_fontsize']);
    plt.subplots_adjust(top = formatting['suptitle_factor']);
    
    if options['show']: plt.show(); 

    return figure