from scdata.utils import std_out
from numpy import arange
from pandas import cut, DataFrame, to_datetime, option_context, to_numeric

'''
Available styles
['_classic_test', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 
'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 
'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 
'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
'''

# markers = ["o" ,"v" ,"^" ,"<" ,">" ,"1" ,"2" ,"3" ,"4" ,"8" ,"s" ,"p" ,
#             "P" , "*" ,"h" ,"H" ,"+" ,"x" ,"X" ,"D" ,"d" ,"|", "_"]

colors = ['sienna',     'gold',
          'orange',   'salmon',    'chartreuse', 'green',      'mediumspringgreen', 'lightseagreen',
          'darkcyan', 'royalblue', 'blue',       'blueviolet', 'purple',            'fuchsia',
          'pink',     'tan',       'olivedrab',  'tomato',     'yellow',            'turquoise']

'''
Colormaps:
Accent, Accent_r, Blues, Blues_r, 
BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, 
Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, 
Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, 
RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, 
Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, 
YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone,
 bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, 
 copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, 
 gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, 
 gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r,
  gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, 
  magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, 
  plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, 
  spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, 
  tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, 
  viridis, viridis_r, vlag, vlag_r, winter, winter_r
'''

# TODO Remove
def export(figure, path, filename = 'plot', library = 'mpl'):
    '''
    Exports a figure to the local filesystem
    Parameters
    ----------
        figure: matplotlib figure or plotly
        path: 'string'
            Path to export it to
        filename: string
            'plot'
            Name for the file to export
        library: string
            'mpl'
            'mpl' (matplotlib) or plotly
    Returns
    ----------
        None

    '''
    
    try:
        std_out('Exporting {} to {}'.format(filename, path))
    
        if library == 'mpl': 
            figure.savefig(join(path, filename + '.png'), dpi = 300, 
                            transparent=False, bbox_inches='tight')
        elif library == 'plotly': 
            pio.write_json(figure, join(path, filename + '.plotly'))
    
    except:
        std_out('Error while exporting, review path', 'ERROR')
        pass
        return
    else:
        std_out('Saved plot', 'SUCCESS')

def prepare_data(test, traces, options):

    std_out('Preparing data for plot')

    # Dataframe to return
    df = DataFrame()
        
    # Check if there are different subplots
    n_subplots = 1

    for trace in traces:
        if 'subplot' in traces[trace].keys(): 
            n_subplots = max(n_subplots, traces[trace]['subplot'])
        else: 
            std_out (f'Trace {trace} not assigned to subplot. Skipping', 'WARNING')

    std_out (f'Making {n_subplots} subplots')
    
    # Generate list of subplots
    subplots = [[] for x in range(n_subplots)]
    
    # Put data in the df
    for trace in traces.keys():

        if 'subplot' not in traces[trace].keys():
            std_out(f'The trace {traces[trace]} was not placed in any subplot. Assuming subplot #1', 'WARNING')
            traces[trace]['subplot'] = 1

        ndevs = traces[trace]['devices']
        channel = traces[trace]['channel']

        if ndevs == 'all': devices = list(test.devices.keys())
        elif type(ndevs) == str or type(ndevs) == int: devices = [ndevs]
        else: devices = ndevs

        for device in devices:

            ndev = str(device)
            
            # Check if device is in columns
            if channel not in test.devices[ndev].readings.columns:
                std_out(f'The device {ndev} does not contain {channel}. Ignoring', 'WARNING')
                continue

            # Put channel in subplots
            subplots[traces[trace]['subplot']-1].append(channel + '_' + ndev)

            column_orig = [channel]
            columns_add = [channel + '_' + ndev]

            # Add filtering name to dfdev
            if 'filter' in traces[trace]:
                col_name = traces[trace]['filter']['col']

                if col_name not in test.devices[ndev].readings.columns:
                    std_out(f'Column {col_name} not in dataframe. Ignoring filtering', 'WARNING')
                else:
                    column_orig.append(col_name)
                    columns_add.append(col_name)
            
            # Device dataframe
            dfdev = DataFrame(test.devices[ndev].readings[column_orig].values,
                            columns = columns_add,
                            index = test.devices[ndev].readings.index)

            # Add filtering function
            if 'filter' in traces[trace]:
                value = traces[trace]['filter']['value']
                relationship = traces[trace]['filter']['relationship']

                if col_name in dfdev.columns:
                    if relationship == '==':
                        dfdev.loc[dfdev[col_name]==value]
                    elif relationship == '<=':
                        dfdev.loc[dfdev[col_name]<=value]
                    elif relationship == '>=':
                        dfdev.loc[dfdev[col_name]>=value]
                    elif relationship == '<':
                        dfdev.loc[dfdev[col_name]<value]
                    elif relationship == '>':
                        dfdev.loc[dfdev[col_name]>value]
                    else:
                        std_out(f"Not valid relationship. Valid options: '==', '<=', '>=', '<', '>'", 'ERROR')
                        continue
                    # Remove column for filtering from dfdev
                    dfdev.drop(columns=[col_name], inplace = True)

            # Combine it in the df
            df = df.combine_first(dfdev)

        # Add average or other extras
        # TODO Check this to simplify
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.resample.Resampler.aggregate.html
        if 'extras' in traces[trace]:
            for extra in traces[trace]['extras']:
                
                extra_name = channel + f'-{extra.upper()}'
                sbl = subplots[traces[trace]['subplot']-1]
                
                if extra == 'max':
                    df[extra_name] = df.loc[:, sbl].max(axis = 1)
                
                if extra == 'mean':
                    df[extra_name] = df.loc[:, sbl].mean(axis = 1)

                if extra == 'min':
                    df[extra_name] = df.loc[:, sbl].min(axis = 1)

                subplots[traces[trace]['subplot']-1].append(extra_name)

    # Trim data
    if options['min_date'] is not None: df = df[df.index > options['min_date']]
    if options['max_date'] is not None: df = df[df.index < options['max_date']]

    # Make sure everything is numeric before resampling
    # https://stackoverflow.com/questions/34257069/resampling-pandas-dataframe-is-deleting-column#34270422
    df = df.apply(to_numeric, errors='coerce')

    # Resample it
    if options['frequency'] is not None: 
        std_out(f"Resampling at {options['frequency']}", "INFO")

        if 'resample' in options: 

            if options['resample'] == 'max': df = df.resample(options['frequency']).max()
            if options['resample'] == 'min': df = df.resample(options['frequency']).min()
            if options['resample'] == 'mean': df = df.resample(options['frequency']).mean()

        else: df = df.resample(options['frequency']).mean()


    # Clean na
    if options['clean_na'] is not None:
        if options['clean_na'] == 'fill':
            df = df.fillna(method='ffill')
        if options['clean_na'] == 'drop':              
            df.dropna(axis = 0, how='any', inplace = True)

    if df.empty: std_out('Dataframe for selected options is empty', 'WARNING')

    return df, subplots

def groupby_session(dataframe, **kwargs):
    '''
    Prepares datafram with groupby options for plots in heatmap or boxplots.
    Parameters
    ----------
        dataframe: pd.DataFrame() 
            Containing the channel and index to groupby.
        frequency_hours: int 
            6
            Bin length in hours
        # TODO - do generic for not only hours
        'periods':
            Periods (before and after) in the form of. There should +1 more dates than labels.
                "periods": {"dates": ['2020-01-20', '2020-03-15', None],
                            "labels": ["Pre-lockdown", "Post-lockdown"]
                            },
    Returns
    -------
        Dataframe with column called 'session'
        Labels for each bin
        The name for the bin category (label_cat)
        Channel name
    '''

    df = dataframe.copy()

    # Add period label
    if 'periods' in kwargs: periods = kwargs['periods']
    else: periods = None

    if periods is not None:

        pdates = periods['dates']

        if pdates[0] is None: pdates[0] = df.index[0]
        if pdates[-1] is None: pdates[-1] = df.index[-1]
        pdates = to_datetime(pdates, utc=True)

        plabels = periods['labels']

        df = df.assign(period = cut(df.index, pdates, labels = plabels, right = False))

    # Assign frequency in hours (TODO)
    if 'frequency_hours' in kwargs: freq_time = kwargs['frequency_hours']
    else: freq_time = 6

    # Include categorical variable
    if freq_time == 6:
        labels = ['Morning','Afternoon','Evening', 'Night']
        label_cat = ''
    elif freq_time == 12:
        labels = ['Morning', 'Evening']
        label_cat = ''
    else:
        labels = [f'{i}h-{i+freq_time}h' for i in arange(0, 24, freq_time)]
        label_cat = 'Hour'

    channel = df.columns[0]
    df = df.assign(session = cut(df.index.hour, arange(0, 25, freq_time), labels = labels, right = False))
    
    # Group them by session
    df_session = df.groupby(['session']).mean()
    df_session = df_session[channel]

    ## Full dataframe
    list_all = ['session', channel]
    
    if periods is not None: list_all.append('period')

    # TODO    
    # # Check relative measurements
    # if self.options['relative']:
    #     # Calculate average
    #     df_session_avg = df_session.mean(axis = 0)
    #     channel = channel + '_REL'

    #     list_all = list(df.columns)
    #     for column in df.columns:
    #         if column != 'session':
    #             df[column + '_REL'] = df[column]/df_session_avg
    #             list_all.append(column + '_REL')
        
    ## Full dataframe
    df = df[list_all]
    df.dropna(axis = 0, how='all', inplace = True)

    return df, labels, label_cat, channel