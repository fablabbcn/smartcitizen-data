from src.visualization.visualization_tools import *
from src.saf import std_out

'''
Colormap PuOr_2 is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, 
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

class Plot(object):
    '''
        This is a plot wrapper for matplotlib (also seaborn) and plotly, that ingests a plot description 
        in json format (or python dict) and a Data object. The main method is plot, shown below.
        Parameters
        ----------
            description: dict
                Dictionary containing the information to make the plot.
                type: Defines the type of the plot. Current available types are:
                    - timeseries
                    - scatter_matrix
                    - correlation_plot
                    - heatmap
                    - barplot
                    - coherence_plot
                    - violin
                    - correlation_comparison
                    - timeseries_comparison
                    - boxplot_comparison
                library: Defines the library to use. Current available types are:
                    - matplotlib
                    - plotly
                TODO: ADD all parameters
        Returns
        -------
            None. Shows plot if requested
    '''
    
    def __init__(self, description):
        # Parse input library
        self.library = description['library']
        if self.library not in ['matplotlib', 'plotly']: raise SystemError ('Not supported library')
        # Plot type
        self.type = description['type']
        # Data
        if 'data' in description.keys(): self.data = description['data']
        # Options
        self.options = description['options']
        # Formatting
        self.formatting = description['formatting']
        if 'style' in self.formatting.keys(): style.use(self.formatting['style'])
        if 'palette' in self.formatting.keys(): sns.set_palette(self.formatting['palette'])
        # Internals
        self.df = pd.DataFrame()
        self.subplots_list = None
        self.figure = None
        
    def prepare_data(self, data):
        std_out('Preparing data for plot')
        if "use_preprocessed" in self.options.keys():
            if self.options["use_preprocessed"]: data_key = 'data_preprocessed'
            else: data_key = 'data'
            
        # Check if there are different subplots
        n_subplots = 1
        for trace in self.data['traces']:
            if 'subplot' in self.data['traces'][trace].keys(): 
                n_subplots = max(n_subplots, self.data['traces'][trace]['subplot'])
                
            else: raise SystemError ('Trace not assigned to subplot')
        
        std_out('Making {} subplots'.format(n_subplots))
        
        # Generate list of subplots
        self.subplots_list = [[] for x in range(n_subplots)]

        # Get min_date and max_date
        if 'min_date' in self.options.keys(): min_date = self.options['min_date']
        else: min_date = None
        if 'max_date' in self.options.keys(): max_date = self.options['max_date']
        else: max_date = None
        
        # Put data in the df
        test = self.data['test']
        for trace in self.data['traces'].keys():
            devicens = self.data['traces'][trace]['device']
            channel = self.data['traces'][trace]['channel']
            
            if devicens != 'all':
                # Put it in list
                if type(devicens) == str or type(devicens) == int: devicens = [devicens]
                for devicen in devicens:
                    if channel not in data.tests[test].devices[str(devicen)].readings.columns: continue
                    # Put channel in subplots_list
                    self.subplots_list[self.data['traces'][trace]['subplot']-1].append(channel + '_' + str(devicen))
                    # Dataframe
                    df = pd.DataFrame(data.tests[test].devices[str(devicen)].readings[channel].values, 
                                      columns = [channel + '_' + str(devicen)],
                                      index = data.tests[test].devices[str(devicen)].readings.index)
                    
                    # Combine it in the df
                    self.df = self.df.combine_first(df)
            
            else:
                for device in data.tests[test].devices.keys():
                    if channel in data.tests[test].devices[device].readings.columns:
                        # Put channel in subplots_list
                        self.subplots_list[self.data['traces'][trace]['subplot']-1].append(channel + '_' + str(device))
                        # Dataframe
                        df = pd.DataFrame(data.tests[test].devices[device].readings[channel].values, 
                                          columns = [channel + '_' + str(device)],
                                          index = data.tests[test].devices[device].readings.index)

                    # Combine it in the df
                    self.df = self.df.combine_first(df)

            # TODO Check this to simplify
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.resample.Resampler.aggregate.html
            if 'include_avg' in self.data['traces'][trace]: 
                avgn = channel + '_TRACEAVG'
                # Make average only of traced channels
                self.df[avgn] = self.df.loc[:, self.subplots_list[self.data['traces'][trace]['subplot']-1]].mean(axis = 1)

                self.subplots_list[self.data['traces'][trace]['subplot']-1].append(avgn)
            
            if 'include_max' in self.data['traces'][trace]: 
                maxn = channel + '_TRACEMAX'
                # Make average only of traced channels
                self.df[maxn] = self.df.loc[:, self.subplots_list[self.data['traces'][trace]['subplot']-1]].max(axis = 1)

                self.subplots_list[self.data['traces'][trace]['subplot']-1].append(maxn)
            
            if 'include_min' in self.data['traces'][trace]: 
                minn = channel + '_TRACEMIN'
                # Make average only of traced channels
                self.df[minn] = self.df.loc[:, self.subplots_list[self.data['traces'][trace]['subplot']-1]].min(axis = 1)
                        
                self.subplots_list[self.data['traces'][trace]['subplot']-1].append(minn)
                
                # if self.data['traces'][trace]['show_all']: 
                #   self.subplots_list[self.data['traces'][trace]['subplot']-1].append(avgn)
                # else:
                #   self.subplots_list[self.data['traces'][trace]['subplot']-1] = [avgn]

        # Trim data
        if min_date is not None: self.df = self.df[self.df.index > min_date]
        if max_date is not None: self.df = self.df[self.df.index < max_date]
            
        # Resample it
        if self.options['frequency'] is not None: 
            if 'resample' in self.options: 
                if self.options['resample'] == 'max': self.df = self.df.resample(self.options['frequency']).max()
                if self.options['resample'] == 'min': self.df = self.df.resample(self.options['frequency']).min()
                if self.options['resample'] == 'mean': self.df = self.df.resample(self.options['frequency']).mean()
            else: self.df = self.df.resample(self.options['frequency']).mean()

        if self.options['clean_na'] is not None:
            if self.options['clean_na'] == 'fill':
                self.df = self.df.fillna(method='ffill')
            if self.options['clean_na'] == 'drop':              
                self.df.dropna(axis = 0, how='any', inplace = True)


    def clean(self):
        # Clean matplotlib cache
        plt.clf()

    def export(self):
        savePath = self.options['export_path']
        fileName = self.options['file_name']
        try:
            std_out('Exporting {} to {}'.format(fileName, savePath))
        
            if self.library == 'matplotlib': self.figure.savefig(savePath+ '/' + fileName + '.png', dpi = 300, transparent=False, bbox_inches='tight')
            elif self.library == 'plotly': pio.write_json(self.figure, savePath+ '/' + fileName + '.plotly')
        except:
            std_out('Error while exporting, review path', 'ERROR')
            pass
            return
        else:
            std_out('Saved plot', 'SUCCESS')

    def groupby_session(self):

        '''
            Prepares data for plots in heatmap or boxplots
        '''

        if 'frequency_hours' in self.formatting.keys(): freq_time = self.formatting['frequency_hours']
        else: freq_time = 6

        # Include categorical variable
        if freq_time == 6:
            labels = ['Morning','Afternoon','Evening', 'Night']
            yaxis = ''
        elif freq_time == 12:
            labels = ['Morning', 'Evening']
            yaxis = ''
        else:
            labels = [f'{i}h-{i+freq_time}h' for i in np.arange(0, 24, freq_time)]
            yaxis = 'Hour'

        channel = self.df.columns[0]
        self.df = self.df.assign(session = pd.cut(self.df.index.hour, np.arange(0, 25, freq_time), labels = labels, right = False))
        
        # Group them by session
        df_session = self.df.groupby(['session']).mean()
        df_session = df_session[channel]

        ## Full dataframe
        list_all = ['session', channel]
        if 'periods' in self.options: list_all.append('period')
        
        # Check relative measurements
        if self.options['relative']:
            # Calculate average
            df_session_avg = df_session.mean(axis = 0)
            channel = channel + '_REL'

            list_all = list(self.df.columns)
            for column in self.df.columns:
                if column != 'session':
                    self.df[column + '_REL'] = self.df[column]/df_session_avg
                    list_all.append(column + '_REL')
            
        ## Full dataframe
        self.df = self.df[list_all]
        self.df.dropna(axis = 0, how='all', inplace = True)

        return labels, yaxis, channel

    # TODO PUT IN @classmethods
    def plot(self, data):

        # Correlation function for plot anotation
        def corrfunc(x, y, **kws):
            r, _ = stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate("r = {:.2f}".format(r),
                        xy=(.1, .9), xycoords=ax.transAxes)

        # Clean matplotlib cache
        plt.clf()
        if 'fontsize' in self.formatting.keys():
            matplotlib.rcParams.update({'font.size': self.formatting['fontsize']})

        # Parse y label and range options if they come from json. Workaround
        if "ylabel" in self.formatting.keys() and self.type != "correlation_plot":
            axises = self.formatting["ylabel"].keys()
            for axis in axises:
                self.formatting["ylabel"][int(axis)] = self.formatting["ylabel"].pop(axis)

        if "yrange" in self.formatting.keys() and self.type != "correlation_plot":
            axises = self.formatting["yrange"].keys()
            for axis in axises:
                self.formatting["yrange"][int(axis)] = self.formatting["yrange"].pop(axis)
        
        # Prepare data for plot
        if data is not None: self.prepare_data(data)
        n_subplots = len(self.subplots_list)

        # Generate plot depending on type and library
        std_out('Plotting')
        
        if self.type == 'timeseries':
            if self.library == 'matplotlib': 
                if self.formatting['width'] > 30: 
                    std_out('Reducing width to 12')
                    self.formatting['width'] = 12
                if self.formatting['height'] > 30: 
                    std_out('Reducing height to 10')
                    self.formatting['height'] = 10
                figure, axes = plt.subplots(n_subplots, 1, sharex = self.formatting['sharex'],
                                                  figsize=(self.formatting['width'], self.formatting['height']));

                if n_subplots == 1:
                    # Check if we are plotting any highlight for the trace
                    if any(['_TRACEAVG' in trace for trace in self.subplots_list[0]]): has_hl = True
                    elif any(['_TRACEMAX' in trace for trace in self.subplots_list[0]]): has_hl = True
                    elif any(['_TRACEMIN' in trace for trace in self.subplots_list[0]]): has_hl = True
                    else: has_hl = False
                    
                    for trace in self.subplots_list[0]:
                        
                        if has_hl:
                            if '_TRACEAVG' in trace: alpha = self.formatting['alpha_highlight']
                            elif '_TRACEMAX' in trace: alpha = self.formatting['alpha_highlight']
                            elif '_TRACEMIN' in trace: alpha = self.formatting['alpha_highlight']
                            else: alpha = self.formatting['alpha_other']
                        else: alpha = 1

                        axes.plot(self.df.index, self.df[trace], label = trace, alpha = alpha)
                        
                        if 'legend' in self.formatting:
                            # TODO make this to compare to not None, so that we can send location
                            if self.formatting['legend']: 
                                axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        axes.set_ylabel(self.formatting['ylabel'][1])
                        
                        if 'xrange' in self.formatting.keys():
                            axes.set_xlim(self.formatting['xrange'][1])
                        if 'yrange' in self.formatting.keys():
                            axes.set_ylim(self.formatting['yrange'][1])

                    if "decorators" in self.formatting:
                        if 'axvline' in self.formatting['decorators']:
                            for vline in self.formatting['decorators']['axvline']:
                                axes.axvline(vline, linestyle = 'dotted', color = 'gray')
                        if 'axhline' in self.formatting['decorators']:
                            for vline in self.formatting['decorators']['axhline']:
                                axes.axhline(vline, linestyle = 'dotted', color = 'gray')                               
                        if 'xtext' in self.formatting['decorators']:
                            for xtext in self.formatting['decorators']['xtext'].keys():
                                text = self.formatting['decorators']['xtext'][xtext]
                                position = self.formatting['yrange'][1][1]-(self.formatting['yrange'][1][1]-self.formatting['yrange'][1][0])/10
                                axes.text(xtext, position, text, size=15, color = 'gray')
                        # TODO Fix
                        if 'ytext' in self.formatting['decorators']:
                            for ytext in self.formatting['decorators']['ytext'].keys():
                                text = self.formatting['decorators']['ytext'][ytext]
                                position = self.formatting['xrange'][1][1]-(self.formatting['xrange'][1][1]-self.formatting['yrange'][1][0])/10
                                axes.text(ytext, position, text, size=15, color = 'gray')                               
                else:
                    for index_subplot in range(n_subplots):
                        for trace in self.subplots_list[index_subplot]:
                            axes[index_subplot].plot(self.df.index, self.df[trace], label = trace)
                            if 'legend' in self.formatting:
                                if self.formatting['legend']: 
                                    axes[index_subplot].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            axes[index_subplot].set_ylabel(self.formatting['ylabel'][index_subplot+1])
                            if 'yrange' in self.formatting.keys():
                                axes[index_subplot].set_ylim(self.formatting['yrange'][index_subplot+1])
                            if 'xrange' in self.formatting.keys():
                                axes[index_subplot].set_xlim(self.formatting['xrange'][index_subplot+1])
                
                _ = figure.suptitle(self.formatting['title'], fontsize=18);
                if 'suptitle_factor' in self.formatting: plt.subplots_adjust(top=self.formatting['suptitle_factor'])
                _ = plt.xlabel(self.formatting['xlabel']);
                _ = plt.grid(self.formatting['grid']);

                # Save it in global and show
                self.figure = figure;
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show();
            
            elif self.library == 'plotly':
                if self.formatting['width'] < 100: 
                    std_out('Setting width to 800')
                    self.formatting['width'] = 800
                if self.formatting['height'] <100: 
                    std_out('Reducing height to 600')
                    self.formatting['height'] = 600             
                figure = make_subplots(rows=n_subplots, cols=1, 
                                       shared_xaxes = self.formatting['sharex'])
                # Add traces
                for index_subplot in range(n_subplots):
                    for trace in self.subplots_list[index_subplot]:
                        figure.append_trace({'x': self.df.index, 
                                             'y': self.df[trace], 
                                             'type': 'scatter', 
                                             'name': trace}, 
                                            index_subplot + 1, 1)
                    # Name the axis
                    figure['layout']['yaxis' + str(index_subplot+1)]['title']['text'] = self.formatting['ylabel'][index_subplot+1]
                    if 'yrange' in self.formatting.keys():
                        figure['layout']['yaxis' + str(index_subplot+1)]['range'] = self.formatting['yrange'][index_subplot+1]

                # Add axis labels
                figure['layout']['xaxis' + str(n_subplots)]['title']['text'] = self.formatting['xlabel']
                
                # Add layout
                figure['layout'].update(height=self.formatting['height'],
                                        legend=dict(x=0.2, y=0, 
                                                    traceorder='normal',
                                                    font = dict(family='sans-serif',
                                                                size=10,
                                                                color='#000'),
                                                    xanchor='center',
                                                    orientation= 'h',
                                                    itemsizing= 'trace',
                                                    yanchor= 'bottom',
                                                    bgcolor='rgba(0,0,0,0)',
                                                    bordercolor='rgba(0,0,0,0)',
                                                    borderwidth = 0),
                                        title=dict(text=self.formatting['title'])
                                       )
                
                self.figure = figure;
                if 'show_plot' in self.options:
                    if self.options['show_plot']: figure.show()

        elif self.type == 'correlation_plot':
            g = sns.jointplot(self.df[self.subplots_list[0][0]], self.df[self.subplots_list[0][1]], 
                          data=self.df, kind=self.formatting['jpkind'], color="b", height=self.formatting['height'])


            # Set title
            g.fig.suptitle(self.formatting['title'])
            # Xlims
            # g.ax_joint.set_xlim(self.formatting['xrange'])
            g.ax_marg_x.set_xlim(self.formatting['xrange'])
            # Ylims
            # g.ax_joint.set_ylim(self.formatting['yrange'])
            g.ax_marg_y.set_ylim(self.formatting['yrange'])

            if self.library == 'plotly':
                plotly_fig = tls.mpl_to_plotly(g.fig)
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plotly_fig.show()
            elif self.library == 'matplotlib':
                # Save it in global and show
                self.figure = g.fig
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show()

            if len(self.subplots_list) > 1: std_out('WARNING: Ignoring additional subplots')

        elif self.type == 'coherence_plot':
            figure = plt.figure(figsize=(self.formatting['width'], self.formatting['height']))

            cxy, f = plt.cohere(self.df[self.subplots_list[0][0]], self.df[self.subplots_list[0][1]], 512, 2, detrend = 'linear')
            plt.ylabel('Coherence [-]')
            # Set title
            figure.suptitle(self.formatting['title'])

            self.figure = figure
            if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show()

            if len(self.subplots_list) > 1: std_out('WARNING: Ignoring additional subplots')

        elif self.type == 'scatter_matrix':

            if self.library == 'matplotlib':
                g = sns.pairplot(self.df.dropna(axis=0, how='all'), vars=self.df.columns[:], height=self.formatting['height'], plot_kws={'alpha': self.formatting['alpha']});
                g.map_upper(corrfunc)
                # g.map_lower(sns.residplot) 
                g.fig.suptitle(self.formatting['title'])
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show();

                self.figure = g.fig;

            elif self.library == 'plotly':

                data = list()
                for column in self.df.columns:
                    data.append(dict(label=column, values=self.df[column]))

                traces = go.Splom(dimensions=data,
                                    marker=dict(color='rgb(0,30,230, 0.35)',
                                    size=5,
                                    #colorscale=pl_colorscaled,
                                    line=dict(width=0.5,
                                              color='rgb(230,230,230, 0.30)')),
                                    diagonal=dict(visible=False))
    
                layout = go.Layout(title=self.formatting['title'],
                            dragmode='zoom',
                            width=self.formatting['width'],
                            height=self.formatting['height'],
                            autosize=False,
                            hovermode='closest',
                            plot_bgcolor='rgba(240,240,240, 0.95)')

                self.figure = dict(data=[traces], layout=layout)
                if 'show_plot' in self.options:
                    if self.options['show_plot']: iplot(self.figure)

        elif self.type == 'heatmap':

            labels, yaxis, channel = self.groupby_session()
            
            if self.library == 'matplotlib':

                # Sample figsize in inches
                _, ax = plt.subplots(figsize=(self.formatting['width'],self.formatting['height']));         
                
                if 'session' in self.options: frequency_session = self.options['session']
                else: frequency_session = '1D'
                xticks = [i.strftime("%Y-%m-%d") for i in self.df.resample(frequency_session).mean().index]
                # Pivot with 'session'
                g = sns.heatmap(self.df.pivot(columns='session').resample(frequency_session).mean().T, ax = ax, \
                    cmap = self.formatting['cmap'], robust = self.formatting['robust'], \
                    vmin = self.formatting['vmin'], vmax = self.formatting['vmax'],
                    #xticklabels = 3,
                    xticklabels = xticks,
                    yticklabels = labels);

                # ax.set_xticks(xticks*ax.get_xlim()[1]/(2))
                _ = g.set_xlabel(self.formatting['xlabel']);
                _ = g.set_ylabel(yaxis);

                # Set title
                _ = g.figure.suptitle(self.formatting['title'], fontsize=18);
                if 'suptitle_factor' in self.formatting: plt.subplots_adjust(top=self.formatting['suptitle_factor'])
                # Save it in global and show
                self.figure = g.figure;
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show()

            elif self.library =='plotly':
                colorscale = [[0, '#edf8fb'], [.3, '#b3cde3'],  [.6, '#8856a7'],  [1, '#810f7c']]

                # Data
                data = [
                    go.Heatmap(
                        z=self.df[channel],
                        x=self.df.index.date,
                        y=self.df['session'],
                        colorscale=colorscale
                    )
                ]

                layout = go.Layout(
                    title=self.formatting['title'],
                    xaxis = dict(ticks=''),
                    yaxis = dict(ticks='' , categoryarray=labels, autorange = 'reversed')
                )

                self.figure = go.Figure(data=data, layout=layout)
                if 'show_plot' in self.options:
                    if self.options['show_plot']: iplot(self.figure)

        elif self.type == 'boxplot':
            
            if 'periods' in self.options:

                pdates = self.options['periods']['dates']

                if pdates[0] is None: pdates[0] = self.df.index[0]
                if pdates[-1] is None: pdates[-1] = self.df.index[-1]
                pdates = pd.to_datetime(pdates, utc=True)

                plabels = self.options['periods']['labels']

                self.df = self.df.assign(period = pd.cut(self.df.index, pdates, labels = plabels, right = False))
            
            labels, xaxis, channel = self.groupby_session()

            if self.library == 'matplotlib':

                # Sample figsize in inches
                _, ax = plt.subplots(figsize=(self.formatting['width'], self.formatting['height']));         
                
                # Pivot with 'session'
                if 'periods' in self.options:
                    g = sns.boxplot(x=self.df['session'], y=self.df[channel], hue = self.df['period'], \
                        ax=ax, palette = self.formatting['cmap']);
                else:
                    g = sns.boxplot(x=self.df['session'], y=self.df[channel], ax=ax, palette = self.formatting['cmap']);
                _ = g.set_ylabel(self.formatting['ylabel'][1]);
                _ = g.set_xlabel(xaxis);
                _ = g.grid(self.formatting['grid']);

                # Set title
                _ = g.figure.suptitle(self.formatting['title']);
                # Save it in global and show
                self.figure = g.figure;
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show()

            elif self.library == 'plotly':
                std_out ('Nothing to see here', 'ERROR')    

        elif self.type == 'violin':

            if self.library == 'matplotlib':

                number_of_subplots = len(self.df.columns) 
                if number_of_subplots % 2 == 0: cols = 2
                else: cols = 3
                rows = int(ceil(number_of_subplots / cols))
                gs = gridspec.GridSpec(rows, cols)
                fig = plt.figure(figsize=(self.formatting['width'], self.formatting['height']*rows/2))

                fig.tight_layout()
                n = 0

                for column in self.df.columns:
                    ax = fig.add_subplot(gs[n])
                    n += 1
                    g = sns.violinplot(y = self.df[column].values, ax = ax, inner="quartile")
                    g.set_xticklabels([column])

                g.figure.suptitle(self.formatting['title'])

                # Save it in global and show
                self.figure = g.figure
                if 'show_plot' in self.options:
                    if self.options['show_plot']: plt.show()
                
            elif self.library == 'plotly':
                std_out ('Nothing to see here', 'ERROR')

        elif self.type == 'correlation_comparison':

            fig = plt.figure( figsize=(self.formatting['width'], self.formatting['height']))

            gs = gridspec.GridSpec(1, 3, figure=fig)
            
            ax1 = fig.add_subplot(gs[0, :-1])
            ax2 = fig.add_subplot(gs[0, -1])
            # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
            print(self.df.columns)
            for index_subplot in range(n_subplots):
                if len(self.subplots_list[index_subplot]) > 2: 
                    std_out('Problem with correlation comparison plot')
                    return
                else:
                    feature_trace = self.subplots_list[index_subplot][0]
                    ref_trace = self.subplots_list[index_subplot][1]
                
                # Calculate basic metrics   
                pearsonCorr = list(self.df.corr('pearson')[list(self.df.columns)[0]])[-1]
                rmse = sqrt(mean_squared_error(self.df[feature_trace].fillna(0), self.df[ref_trace].fillna(0)))
        
                std_out ('Pearson correlation coefficient: ' + str(pearsonCorr))
                std_out ('Coefficient of determination R²: ' + str(pearsonCorr*pearsonCorr))
                std_out ('RMSE: ' + str(rmse))
                std_out ('')
                
                # Time Series plot
                ax1.plot(self.df.index, self.df[feature_trace], color = 'g', label = feature_trace, linewidth = 1, alpha = 0.9)
                ax1.plot(self.df.index, self.df[ref_trace], color = 'k', label = ref_trace, linewidth = 1, alpha = 0.7)
                ax1.axis('tight')
                ax1.set_title('Time Series Plot for {}'.format(self.formatting['title']), fontsize=14)
                ax1.grid(True)
                ax1.legend(loc="best")
                ax1.set_xlabel(self.formatting['xlabel'])
                ax1.set_ylabel(self.formatting['ylabel'][index_subplot+1])
                if 'yrange' in self.formatting.keys():
                    ax1.set_ylim(self.formatting['yrange'][index_subplot+1])
                
                # Correlation plot
                ax2.plot(self.df[ref_trace], self.df[feature_trace], 'go', label = feature_trace, linewidth = 1,  alpha = 0.3)
                ax2.plot(self.df[ref_trace], self.df[ref_trace], 'k', label =  '1:1 Line', linewidth = 0.4, alpha = 0.8)
                ax2.axis('tight')
                ax2.set_title('Scatter Plot for {}'.format(self.formatting['title']), fontsize=14)
                ax1.grid(True)
                ax2.grid(True)
                ax2.legend(loc="best")
                ax2.set_xlabel(self.formatting['ylabel'][index_subplot+1])
                ax2.set_ylabel(self.formatting['ylabel'][index_subplot+1])
                
                if 'yrange' in self.formatting.keys():
                    ax2.set_xlim(self.formatting['yrange'][index_subplot+1])
                    ax2.set_ylim(self.formatting['yrange'][index_subplot+1])
                                # Save it in global and show
            self.figure = fig
            if 'show_plot' in self.options:
                if self.options['show_plot']: plt.show()

        elif self.type == 'timeseries_comparison':
            if 'periods' in self.options: periods = self.options['periods']
            print ('Not yet')

        
            


