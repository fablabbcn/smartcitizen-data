# Numeric
from math import sqrt, ceil, isnan
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from scipy import stats

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import gridspec
plt.ioff()
# Seaborn
import seaborn as sns
# Plotly
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.io as pio
pio.renderers.default = "jupyterlab"
# Folium
from folium import plugins, Map, Marker, Circle
# Other
from traceback import print_exc

'''
Available styles
['_classic_test', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
'''

markers = ["o" ,"v" ,"^" ,"<" ,">" ,"1" ,"2" ,"3" ,"4" ,"8" ,"s" ,"p" ,"P" ,"*" ,"h" ,"H" ,"+" ,"x" ,"X" ,"D" ,"d" ,"|", "_"]
colors = ['black',    'silver',    'red',        'sienna',     'gold',
          'orange',   'salmon',    'chartreuse', 'green',      'mediumspringgreen', 'lightseagreen',
          'darkcyan', 'royalblue', 'blue',       'blueviolet', 'purple',            'fuchsia',
          'pink',     'tan',       'olivedrab',  'tomato',     'yellow',            'turquoise']


def target_diagram(models, plot_train, style_to_use = 'seaborn-paper'):
    ## TODO DOCUMENT
    style.use(style_to_use)
    
    def minRtarget(targetR):
        return sqrt(1+ np.power(targetR,2)-2*np.power(targetR,2))

    targetR20 = 0.5
    targetR0 = sqrt(targetR20)
    MR0 = minRtarget(targetR0)
    targetR21 = 0.7
    targetR1 = sqrt(targetR21)
    MR1 = minRtarget(targetR1)
    targetR22 = 0.9
    targetR2 = sqrt(targetR22)
    MR2 = minRtarget(targetR2)

    fig  = plt.figure(figsize=(13,13))
    i = -1
    prev_group = 0
    for model in models:
        try:
            metrics_model = models[model]

            if prev_group != models[model]['group']: i = 0
            else: i+=1
            
            if models[model]['group'] > len(colors)-1: 
                color_group = colors[models[model]['group']-len(colors)]
            else: 
                color_group = colors[models[model]['group']]
            marker_group = markers[i]
            plt.scatter(metrics_model['sign_sigma']*metrics_model['RMSD_norm_unb'], metrics_model['normalised_bias'], 
                label = model, color = color_group, marker = marker_group, s = 100, alpha = 0.7)
            prev_group = models[model]['group']
        except:
            print_exc()
            print ('Cannot plot model {}'.format(model))
    ## Display and others
    plt.axhline(0, color='gray', linewidth = 0.8)
    plt.axvline(0, color='gray', linewidth = 0.8)

    ## Add circles
    ax = plt.gca()
    circle1 =plt.Circle((0, 0), 1, linewidth = 1.4, color='gray', fill =False)
    circleMR0 =plt.Circle((0, 0), MR0, linewidth = 1.4, color='r', fill=False)
    circleMR1 =plt.Circle((0, 0), MR1, linewidth = 1.4, color='y', fill=False)
    circleMR2 =plt.Circle((0, 0), MR2, linewidth = 1.4, color='g', fill=False)
    
    circle3 =plt.Circle((0, 0), 0.01, color='g', fill=True)
    
    ## Add annotations
    ax.add_artist(circle1)
    ax.annotate('R2 < 0',
                xy=(1, 0), xycoords='data',
                xytext=(-35, 10), textcoords='offset points')
    
    ax.add_artist(circleMR0)
    ax.annotate('R2 < ' + str(targetR20),
                xy=(MR0, 0), xycoords='data',
                xytext=(-35, 10), textcoords='offset points', color = 'r')
    
    ax.add_artist(circleMR1)
    ax.annotate('R2 < ' + str(targetR21),
                xy=(MR1, 0), xycoords='data',
                xytext=(-45, 10), textcoords='offset points', color = 'y')
    
    
    ax.add_artist(circleMR2)
    ax.annotate('R2 < ' + str(targetR22),
                xy=(MR2, 0), xycoords='data',
                xytext=(-45, 10), textcoords='offset points', color = 'g')
    ax.add_artist(circle3)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.title('Target Diagram')
    plt.ylabel('Normalised Bias (-)')
    plt.xlabel("RMSD*'")
    plt.grid(True)
    plt.show()

    return fig

def scatter_diagram(fig, gs, n, dataframeTrain, dataframeTest):
    ## TODO DOCUMENT

    ax = fig.add_subplot(gs[n])

    plt.plot(dataframeTrain['reference'], dataframeTrain['prediction'], 'go', label = 'Train ' + model_name, alpha = 0.3)
    plt.plot(dataframeTest['reference'], dataframeTest['prediction'], 'bo', label = 'Test ' + model_name, alpha = 0.3)
    plt.plot(dataframeTrain['reference'], dataframeTrain['reference'], 'k', label = '1:1 Line', linewidth = 0.4, alpha = 0.3)

    plt.legend(loc = 'best')
    plt.ylabel('Prediction (-)')
    plt.xlabel('Reference (-)')

## TODO Make more generic
def map_defaults(options = dict()):
    # Sets default options for map representation
    # Location to Barcelona
    if 'location' not in options: options['location'] = [41.400818, 2.1825157] 
    if 'tiles' not in options: options['tiles'] = 'Stamen Toner'
    if 'zoom' not in options: options['zoom'] = 2.5
    if 'period' not in options: options['period'] = '1W'
    if 'radius' not in options: options['radius'] = 10
    if 'fillOpacity' not in options: options['fillOpacity'] = 1
    if 'stroke' not in options: options['stroke'] = 'false'
    if 'icon' not in options: options['icon'] = 'circle'

    return options

def device_history_map(map_type = 'dynamic', dataframe = None, options = dict()):
    '''
    Creates a folium map with either location of devices or their "existence period"
    -------    
    Parameters:
    map_type: String
        'dynamic'
        'dynamic' or 'static'. Whether is a dinamic map or not
    dataframe: Pandas Dataframe
        None
        Contains information about when the devices started posting data, ids, location. It follows the format of world_map in api device
    options: dict
        dict()

    Returns:
        Folium.Map object
    '''

    def coordinates(x):
        return [x['latitude'], x['longitude']]

    def color(x):
        iSCAPE_IDs =[19, 20, 21, 28]
        making_sense_IDs = [11, 14]
        SCK_21_IDs = [26]

        color = '#0019ff'

        try:
            if x['kit_id'] in iSCAPE_IDs: color = '#7dbd4c'
            elif x['kit_id'] in making_sense_IDs: color = '#f88027'
            elif x['kit_id'] in SCK_21_IDs: color = '#ffb500'
        except:            
            print_exc()
            pass
            
        return color

    def validate(x):
        
        if x['last_reading_at'] is None: return False
        if x['added_at'] is None: return False
        if any(x['coordinates']) is None or any([isnan(item) for item in x['coordinates']]): return False
        if map_type == 'dynamic':
            if x['date_list'] == []: return False
        
        return True

    def range_list(x):
        
        date_r = pd.date_range(start=x['added_at'], end=x['last_reading_at'], normalize = True, freq=options['period']).strftime('%Y-%m-%d')
        date_l = list()
        for item in date_r.values: date_l.append(str(item))

        return date_l

    dataframe['color'] = dataframe.apply(lambda x: color(x), axis=1)
    dataframe['coordinates'] = dataframe.apply(lambda x: coordinates(x), axis=1)

    options = map_defaults(options)

    if map_type == 'dynamic':
        dataframe['date_list'] = dataframe.apply(lambda x: range_list(x), axis=1)
        dataframe['valid'] = dataframe.apply(lambda x: validate(x), axis=1)

        dataframe = dataframe[(dataframe['valid'] == True)]
        features = list()

        for sensor in dataframe.index:
            features.append(
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [dataframe.loc[sensor, 'coordinates'][::-1]]*len(dataframe.loc[sensor, 'date_list']),
                        'popup': f'<a href="http://smartcitizen.me/kits/{sensor}">{sensor}</a>',
                    },
                    'properties': {
                        'times': dataframe.loc[sensor, 'date_list'],
                        'icon': options['icon'],
                        'iconstyle': {
                            'fillColor': dataframe.loc[sensor, 'color'],
                            'fillOpacity': options['fillOpacity'],
                            'stroke': options['stroke'],
                            'radius': options['radius']
                        },
                        'style': {'weight': '0'},
                        'id': 'man'
                    }
                }
            )

        m = make_map(map_type = map_type, features = features, options = options)

    elif map_type == 'static':
        m = make_map(map_type = map_type, options = options)
        dataframe['valid'] = dataframe.apply(lambda x: validate(x), axis=1)
        dataframe = dataframe[(dataframe['valid'] == True)]

        for sensor in dataframe.index:
            Circle(
                location = dataframe.loc[sensor, 'coordinates'],
                radius=options['radius'],
                color=dataframe.loc[sensor, 'color'],
                fill=True,
                fillOpacity=options['fillOpacity'],
                fillColor=dataframe.loc[sensor, 'color'],
                popup = f'<a href="http://smartcitizen.me/kits/{sensor}">{sensor}</a>'
            ).add_to(m)

    return m

def make_map(map_type = 'dynamic', features = None, options = dict()):
    '''
    Creates a folium map based on already created features and a set of options for customization    
    Parameters
    -------
    map_type: String
        'dynamic'
        'dynamic' or 'static'. Whether is a dinamic map or not
    features: JSON iterable
        None
        JSON format for folium map features
    options: dict
        dict()
    Returns
    -------    
        Folium.Map object
    '''

    m = Map(
        location=options['location'],
        tiles=options['tiles'],
        zoom_start=options['zoom'],
    )

    if map_type == 'static': return m

    plugins.TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period='P'+options['period'],
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=5,
        loop_button=True,
        date_options='YYYY/MM/DD',
        time_slider_drag_update=True,
        duration='P'+options['period']
    ).add_to(m)

    return m

def device_metric_map(test, channel, start_date = None, end_date = None, options = dict()):
    '''
    Creates a folium map showing the evolution of a metric dynamically with colors
    Parameters
    -------
    test: Test
        Test object containing devices
    channel: String
        The channel to make the map from
    start_date, end_date: String
        None
        Date convertible string
    options: dict()
        dict()
        See map_defaults for options
    Returns
    -------    
        Folium.Map object
    '''

    # Map color bins
    poll_colors_palette = np.array(['#053061','#2166ac','#4393c3','#92c5de','#d1e5f0','#fddbc7','#f4a582','#d6604d','#b2182b','#67001f'])

    channel_bins = {
        'NOISE': [-np.inf, 52, 54, 56, 58, 60, 62, 64, 66, 68, np.inf],
        'PM': [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, np.inf]
    }

    # Set defaults
    options = map_defaults(options)

    # Make date range
    date_r = pd.date_range(start=start_date, end=end_date, normalize = True, freq=options['period']).strftime('%Y-%m-%d')
    date_l = list()
    for item in date_r.values: date_l.append(str(item))

    # Get bins
    for bname in channel_bins.keys():
        if bname in channel: bins = channel_bins[bname]; break

    # Make features
    features = []
    for device in test.devices:
        # Get lat, long
        lat = test.devices[str(device)].api_device.lat
        long = test.devices[str(device)].api_device.long
        if lat is None or long is None: continue
        # Resample
        try:
            dfc = test.devices[str(device)].readings.resample(options['period']).mean()    
        except:
            pass
            continue

        if channel not in dfc.columns: continue
        # Make color column
        dfc['color'] = pd.cut(dfc[channel], bins, labels=poll_colors_palette)

        # Add point for each date
        for date in date_l:
            if date not in dfc.index: continue
            if date_l.index(date) > len(date_l)-2: continue
            features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[str(long), str(lat)]]*2,
                        'popup': str(device),
                    },
                    'properties': {
                        'times': [date, date_l[date_l.index(date)+1]],
                        'icon': options['icon'],
                        'iconstyle': {
                            'fillColor': str(dfc.loc[date,'color']),
                            'fillOpacity': options['fillOpacity'],
                            'stroke': options['stroke'],
                            'radius': options['radius']
                        },
                        'style': {'weight': '0'},
                        'id': 'man'
                    }
                }
            )

    # Make map
    m = make_map(map_type = 'dynamic', features = features, options = options)

    return m