''' Device metric map implementation '''

from folium import Map, Marker, Circle, plugins
from math import isnan
from traceback import print_exc
from math import inf
from pandas import cut, date_range
from numpy import array
from scdata.utils import dict_fmerge
from scdata._config import config

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
        Contains information about when the devices started posting data, ids, location. 
        It follows the format of world_map in api device
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
        
        date_r = date_range(start=x['added_at'], end=x['last_reading_at'], normalize = True, freq=options['period']).strftime('%Y-%m-%d')
        date_l = list()
        for item in date_r.values: date_l.append(str(item))

        return date_l

    dataframe['color'] = dataframe.apply(lambda x: color(x), axis=1)
    dataframe['coordinates'] = dataframe.apply(lambda x: coordinates(x), axis=1)

    options = dict_fmerge(config._map_def_opt, options)

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
                            'fillOpacity': options['fillOpacity'],
                            'fillColor': dataframe.loc[sensor, 'color'],
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


def device_metric_map(self, channel, start_date, end_date, options = dict()):
    '''
    Creates a folium map showing the evolution of a metric dynamically with colors
    Parameters
    -------
    channel: String
        The channel to make the map from
    start_date, end_date: String
        Date convertible string
    options: dict()
        Possible keys are (default otherwise)
            location: list
                [41.400818, 2.1825157] 
                Center map location
            tiles: (String)
                'Stamen Toner'
                Tiles for the folium.Map
            zoom: (float)
                2.5
                Zoom to start with in folium.Map
            period: 'String'
                '1W'
                Period for 'dynamic' map
            radius: float 
                10
                Circle radius for icon
            fillOpacity: float
                1
                (<1) Fill opacity for the icon
            stroke: 'String'
                'false'
                'true' or 'false'. For icon's stroke
            icon: 'String' 
                'circle'
                A valid folium.Map icon style
    Returns
    -------    
        Folium.Map object
    '''

    # Map color bins
    poll_colors_palette = array(['#053061','#2166ac','#4393c3','#92c5de','#d1e5f0','#fddbc7','#f4a582','#d6604d','#b2182b','#67001f'])

    channel_bins = {
        'NOISE': [-inf, 52, 54, 56, 58, 60, 62, 64, 66, 68, inf],
        'PM': [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, inf]
    }

    # Set defaults
    options = dict_fmerge(config._map_def_opt, options)

    # Make date range
    date_r = date_range(start=start_date, end=end_date, normalize = True, freq=options['period']).strftime('%Y-%m-%d')
    date_l = list()
    for item in date_r.values: date_l.append(str(item))

    # Get bins
    for bname in channel_bins.keys():
        if bname in channel: bins = channel_bins[bname]; break

    # Make features
    features = []
    for device in self.devices:

        # Get lat, long
        try:
            self.devices[str(device)].api_device.get_device_lat_long()
            lat = self.devices[str(device)].api_device.lat
            long = self.devices[str(device)].api_device.long
        except AttributeError:
            pass 
            continue

        if lat is None or long is None: continue
        
        # Resample
        try:
            dfc = self.devices[str(device)].readings.resample(options['period']).mean()    
        except:
            pass
            continue

        if channel not in dfc.columns: continue
        # Make color column
        dfc['color'] = cut(dfc[channel], bins, labels = poll_colors_palette)

        # Add point for each date
        for date in date_l:
            if date not in dfc.index: continue
            if date_l.index(date) > len(date_l) - 2: continue
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