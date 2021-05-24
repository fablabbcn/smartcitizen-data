''' Device metric map implementation '''

from folium import (Map, Marker, Circle, GeoJson, DivIcon,
                    GeoJsonTooltip, GeoJsonPopup)

from folium.plugins import MiniMap, TimestampedGeoJson
from math import isnan, floor, ceil
from traceback import print_exc
from pandas import cut, date_range
from scdata.utils import dict_fmerge, clean, std_out
from scdata._config import config
from numpy import linspace, nan
from branca import element
from jinja2 import Template, FileSystemLoader, Environment
import json
from os.path import dirname, join

def convert_rollup(frequency):
    ''' Converts pandas to folium period '''

    pandas_folium_period_convertion = (
                                    ['S','S'],
                                    ['Min','M'],
                                    ['H','H'],
                                    ['D','D'],
                                    ['W','W'],
                                    ['M',''],
                                    ['Y','']
                                    )

    # Convert frequency from pandas to API's
    for index, letter in enumerate(frequency):
        try:
            aux = int(letter)
        except:
            index_first = index
            letter_first = letter
            rollup_value = frequency[:index_first]
            frequency_unit = frequency[index_first:]
            break

    for item in pandas_folium_period_convertion:
        if item[0] == frequency_unit:
            rollup_unit = item[1]
            break

    rollup = rollup_value + rollup_unit
    return rollup

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

        date_r = date_range(start=x['added_at'], end=x['last_reading_at'],
                            normalize = True, freq=options['period']).strftime('%Y-%m-%d')
        date_l = list()
        for item in date_r.values: date_l.append(str(item))

        return date_l

    dataframe['color'] = dataframe.apply(lambda x: color(x), axis=1)
    dataframe['coordinates'] = dataframe.apply(lambda x: coordinates(x), axis=1)

    options = dict_fmerge(config._map_def_opt, options)

    # Make map
    m = Map(
        location=options['location'],
        tiles=options['tiles'],
        zoom_start=options['zoom'],
    )

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

        TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': features
            },
            period='P'+convert_rollup(options['period']),
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=options['max_speed'],
            loop_button=True,
            # date_options='YYYY/MM/DD',
            time_slider_drag_update=True,
            duration='P'+options['period']
        ).add_to(m)

    elif map_type == 'static':

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

    # Set defaults
    options = dict_fmerge(config._map_def_opt, options)

    # Make date range
    date_r = date_range(start=start_date, end=end_date, normalize = True,
                        freq=options['period']).strftime('%Y-%m-%d')
    date_l = list()
    for item in date_r.values: date_l.append(str(item))

    # Get bins
    for bname in config._channel_bins.keys():
        if bname in channel: bins = config._channel_bins[bname]; break

    # Make features
    features = []
    for device in self.devices:

        # Get lat, long
        try:
            self.devices[str(device)].api_device.get_device_lat_long()
            _lat = self.devices[str(device)].api_device.lat
            _long = self.devices[str(device)].api_device.long
        except AttributeError:
            std_out(f'Cannot retrieve [lat, long] from device {device}', 'ERROR')
            pass
            continue

        if _lat is None or _long is None: continue

        # Resample
        try:
            dfc = self.devices[str(device)].readings.resample(options['period']).mean()
        except:
            pass
            continue

        if channel not in dfc.columns: continue
        # Make color column
        dfc['color'] = cut(dfc[channel], bins, labels = config._map_colors_palette)

        # Add point for each date
        for date in date_l:
            if date not in dfc.index: continue
            if date_l.index(date) > len(date_l) - 2: continue
            features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[str(_long), str(_lat)]]*2,
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
    m = Map(
        location=options['location'],
        tiles=options['tiles'],
        zoom_start=options['zoom'],
    )

    TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period='P'+convert_rollup(options['period']),
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=5,
        loop_button=True,
        # date_options='YYYY/MM/DD',
        time_slider_drag_update=True,
        duration='P'+options['period']
    ).add_to(m)

    return m

def path_plot(self, channel = None, map_type = 'dynamic', devices = 'all',
              start_date = None, end_date = None, options = dict()):
    '''
    Creates a folium map showing a path
    Parameters
    -------
    channel: String
        None
        If None, shows path, otherwise, colored path with channel mapping
    map_type: String
        'dynamic'
        'dynamic' or 'static'. Whether is a dinamic map or not
    devices: list or 'all'
        List of devices to include, or 'all' from self.devices
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

    # Set defaults
    options = dict_fmerge(config._map_def_opt, options)

    # Make features
    features = []
    if devices == 'all':
        mdev = self.devices
    else:
        mdev = list()
        for device in devices:
            if device in self.devices: mdev.append(device)
            else: std_out(f'Device {device} not found, ignoring', 'WARNING')

    if len(mdev) == 0:
        std_out('Requested devices not in test', 'ERROR')
        return None

    for device in mdev:
        chs = ['GPS_LAT','GPS_LONG']
        if channel is not None:
            if channel not in self.devices[str(device)].readings.columns:
                std_out(f'Channel {channel} not in columns: {self.devices[str(device)].readings.columns}', 'ERROR')
                return None

            # Get bins
            minmax = False
            if not options['minmax']:
                if all([key not in channel for key in config._channel_bins]):
                    std_out(f'Requested channel {channel} not in config mapped bins {config._channel_bins.keys()}.Using min/max mapping', 'WARNING')
                    minmax = True
            else:
                minmax = True

            if minmax:
                bins = linspace(self.devices[str(device)].readings[channel].min(),
                    self.devices[str(device)].readings[channel].max(),
                    config._channel_bin_n)
            else:
                for bname in config._channel_bins.keys():
                    if bname in channel: bins = config._channel_bins[bname]; break
            chs.append(channel)

        # Create copy
        dfc = self.devices[str(device)].readings[chs].copy()
        # Resample and cleanup
        # TODO THIS CAN INPUT SOME MADE UP READINGS
        dfc = clean(dfc.resample(options['period']).mean(), 'fill')

        # Make color column
        legend_labels = None
        if channel is not None:
            dfc['COLOR'] = cut(dfc[channel], bins, labels =\
                config._map_colors_palette)

                # Make legend labels
            legend_labels = {}
            for ibin in range(len(bins)-1):
                legend_labels[f'{round(bins[ibin],2)} : {round(bins[ibin+1],2)}'] =\
                    config._map_colors_palette[ibin]
        else:
            dfc['COLOR'] = config._map_colors_palette[0]

        if start_date is not None:
            dfc = dfc[dfc.index>start_date]
        if end_date is not None:
            dfc = dfc[dfc.index<end_date]

        # Add point for each date
        for date in dfc.index:
            if date == dfc.index[-1]: break
            times = []

            color = str(dfc.loc[date, 'COLOR'])
            if color == 'nan' or isnan(dfc.loc[date, 'GPS_LONG'])\
            or isnan(dfc.loc[date, 'GPS_LAT']):
                std_out(f'Skipping point {date}', 'WARNING'); continue

            geometry = {
                'type': 'LineString',
                'coordinates': [[dfc.loc[date, 'GPS_LONG'],
                    dfc.loc[date, 'GPS_LAT']],
                    [dfc.loc[date+dfc.index.freq, 'GPS_LONG']
                    ,dfc.loc[date+dfc.index.freq, 'GPS_LAT']]],
                }

            properties = {
                'icon': options['icon'],
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': options['fillOpacity'],
                    'stroke': options['stroke'],
                    'radius': options['radius']
                },
                'device': device,
                'timestamp': date.strftime('%Y-%m-%dT%H:%M:%S'),
                "coordinates": [dfc.loc[date+dfc.index.freq, 'GPS_LAT']
                    ,dfc.loc[date+dfc.index.freq, 'GPS_LONG']],
                'style': {
                    'color': color,
                    'stroke-width': options['stroke-width'],
                    'fillOpacity': options['fillOpacity']
                }
            }

            # Add reading to tooltip
            if channel is not None:
                properties['channel'] = channel
                properties['value'] = dfc.loc[date, channel]

            if map_type == 'dynamic':
                properties['times'] = [date.strftime('%Y-%m-%dT%H:%M:%S'),
                    (date + dfc.index.freq).strftime('%Y-%m-%dT%H:%M:%S')]

            features.append({
                'type': 'Feature',
                'geometry': geometry,
                'properties': properties
            })

    featurecol = {
        'type': 'FeatureCollection',
        'features': features
    }

    # Make map
    if options['location'] == 'average':
        avg_long = dfc['GPS_LONG'].mean()
        avg_lat  = dfc['GPS_LAT'].mean()
        loc = [avg_lat, avg_long]
    else:
        loc = options['location']

    m = Map(
        location=loc,
        tiles=options['tiles'],
        zoom_start=options['zoom'],
    )

    if map_type == 'static':
        # TODO WORKAROUND UNTIL GEOJSON ACCEPTS MARKERS
        if options['markers']:
            for feature in features:
                Circle(
                    location=[feature['geometry']['coordinates'][0][1],
                        feature['geometry']['coordinates'][0][0]],
                    fill='true',
                    radius = feature['properties']['iconstyle']['radius'],
                    color = feature['properties']['iconstyle']['fillColor'],
                    fill_opacity=feature['properties']['iconstyle']['fillOpacity']
                ).add_to(m)

        if channel is not None:
            fields=["device", "channel", "timestamp",
                "coordinates", "value"]
            aliases=["Device:", "Sensor:", "Timestamp:",
                "Coordinates:", "Reading:"]
        else:
            fields=["device", "timestamp", "coordinates"]
            aliases=["Device:", "Timestamp:", "Coordinates:"]

        popup = GeoJsonPopup(
            fields=fields,
            aliases=aliases,
            localize=True,
            labels=True,
            max_width=800,
        )

        tooltip = GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 1px solid gray;
                border-radius: 1px;
                box-shadow: 2px;
            """,
            max_width=800,
        );

        GeoJson(featurecol,
            tooltip=tooltip,
            popup=popup,
            style_function=lambda x: {
                'color': x['properties']['style']['color'],
                'weight': x['properties']['style']['stroke-width'],
                'fillOpacity': x['properties']['style']['fillOpacity']            },
        ).add_to(m)

    elif map_type == 'dynamic':
        TimestampedGeoJson(
            featurecol,
            period='PT'+convert_rollup(options['period']),
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=options['max_speed'],
            loop_button=True,
            time_slider_drag_update=True
        ).add_to(m)

    else:
        std_out(f'Not supported map type {map_type}', 'ERROR')
        return None

    if options['minimap']:
        minimap = MiniMap(toggle_display=True, tile_layer=options['tiles'])
        minimap.add_to(m)

    if options['legend'] and not legend_labels is None:

        templateLoader = FileSystemLoader(searchpath=join(dirname(__file__),\
            'templates'))
        templateEnv = Environment(loader=templateLoader)
        template = templateEnv.get_template("map_legend.html")

        filled_map_legend = template.render(legend_labels=legend_labels)

        map_legend_html = '{% macro html(this, kwargs) %}'+\
            filled_map_legend+\
            '{% endmacro %}'

        legend = element.MacroElement()
        legend._template = element.Template(map_legend_html)

        m.get_root().add_child(legend)

    return m
