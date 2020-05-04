''' Device metric map implementation '''

from .map_tools import map_defaults, make_map
from math import inf
from pandas import cut, date_range
from numpy import array

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
    options = map_defaults(options)

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