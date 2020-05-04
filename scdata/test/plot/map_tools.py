''' Tools for map plotting '''

from folium import Map, Marker, Circle, plugins
from math import isnan
from pandas import date_range
from traceback import print_exc

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
