''' Device metric map implementation '''

import json
from math import ceil, floor, isnan
from os.path import dirname, join
from traceback import print_exc

from branca import element
from folium import (Circle, DivIcon, GeoJson, GeoJsonPopup, GeoJsonTooltip,
                    Map, Marker)
from folium.plugins import MiniMap, TimestampedGeoJson
from jinja2 import Environment, FileSystemLoader, Template
from numpy import linspace, nan
from pandas import Timedelta, Timestamp, concat, cut, date_range
from pandas.tseries.frequencies import to_offset

from scdata._config import config
from scdata.tools.cleaning import clean
from scdata.tools.custom_logger import logger
from scdata.tools.dictmerge import dict_fmerge


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
            period=iso_period(options['period']),
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=options['max_speed'],
            loop_button=True,
            # date_options='YYYY/MM/DD',
            time_slider_drag_update=True,
            duration=iso_period(options['period'])
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

def map_devices(self, devices='all'):
    ''' The devices of a test (all, or those with the ids given), or the device itself '''
    from scdata.device.device import Device
    available = [self] if isinstance(self, Device) else list(self.devices)
    if devices == 'all':
        return available
    ids = {str(device_id) for device_id in devices}
    found = [device for device in available if str(device.id) in ids]
    for device_id in ids - {str(device.id) for device in found}:
        logger.warning(f'Device {device_id} not found, ignoring')
    return found


def iso_period(frequency):
    ''' ISO 8601 duration of a pandas frequency (e.g. 10Min: P0DT0H10M0S), for the time slider '''
    offset = to_offset(frequency)
    try:
        delta = Timedelta(offset)
    except ValueError:
        # Weeks are anchored to a weekday: not a Timedelta for pandas
        if not offset.name.startswith('W'):
            raise ValueError(f'{frequency} is not a fixed period (use days, hours or minutes)')
        delta = Timedelta(weeks=offset.n)
    return delta.isoformat()


def channel_bins(channel, series, minmax=False):
    ''' Bins of the colors: from config._channel_bins, or between the minimum and maximum of the data '''
    if not minmax:
        for name, bins in config._channel_bins.items():
            if name in channel:
                return bins
        logger.warning(f'{channel} not in the bins of config._channel_bins: using its minimum and maximum')
    low, high = series.min(), series.max()
    if low == high:
        # A constant series: bins around its value (cut needs increasing edges)
        low, high = low - 0.5, high + 0.5
    return linspace(low, high, len(config._map_colors_palette) + 1)


def colors_of(series, bins):
    '''
    Color of each value: the lowest edge included, and values outside the bins (above or below
    config._channel_bins) with the color of the closest bin. Missing values stay NaN
    '''
    return cut(series.clip(bins[0], bins[-1]), bins, labels=config._map_colors_palette, include_lowest=True)


def legend_labels(bins):
    return {f'{round(bins[index], 2)} : {round(bins[index + 1], 2)}': config._map_colors_palette[index]
            for index in range(len(bins) - 1)}


def add_legend(m, labels):
    template = Environment(loader=FileSystemLoader(searchpath=join(dirname(__file__), 'templates')))\
        .get_template('map_legend.html')
    legend = element.MacroElement()
    legend._template = element.Template('{% macro html(this, kwargs) %}' + template.render(legend_labels=labels)
                                        + '{% endmacro %}')
    m.get_root().add_child(legend)


def in_period(dataframe, start_date, end_date):
    ''' Rows in [start_date, end_date), dates in the time zone of the data when they have none '''
    for date, keep in ((start_date, lambda index, value: index >= value), (end_date, lambda index, value: index < value)):
        if date is None:
            continue
        date = Timestamp(date)
        if dataframe.index.tz is not None and date.tzinfo is None:
            date = date.tz_localize(dataframe.index.tz)
        dataframe = dataframe[keep(dataframe.index, date)]
    return dataframe


def device_metric_map(self, channel, start_date=None, end_date=None, devices='all', options=dict()):
    '''
    Creates a folium map with the devices at their location, colored by the average of a channel in each
    period, with a time slider
    Parameters
    -------
    channel: String
        The channel to make the map from
    start_date, end_date: String
        None
        Date convertible strings (the whole data by default)
    devices: list or 'all'
        Ids of the devices of a test to include
    options: dict()
        Possible keys are (defaults in config._map_def_opt)
            location: list or 'average'
                Center of the map, or the average location of the devices
            tiles: String
                Tiles for the folium.Map
            zoom: float
                Zoom to start with
            period: String
                '1W'
                Period of the averages (pandas frequency)
            minmax: bool
                False
                Colors between the minimum and maximum of the data, instead of config._channel_bins
            radius, fillOpacity, stroke, icon, max_speed, legend
    Returns
    -------
        folium.Map
    '''
    options = dict_fmerge(config._map_def_opt, options)

    features, locations, values = [], [], []
    for device in map_devices(self, devices):
        location = getattr(device.handler.json, 'location', None) if device.handler is not None else None
        if location is None or location.latitude is None or location.longitude is None:
            logger.warning(f'Device {device.id} has no location')
            continue
        if channel not in device.data.columns:
            logger.warning(f'{channel} not in device {device.id}')
            continue
        series = in_period(device.data[[channel]], start_date, end_date)[channel].resample(options['period']).mean()
        if series.dropna().empty:
            continue
        locations.append([location.latitude, location.longitude])
        values.append((device, location, series))

    if not values:
        logger.error(f'No device with location and {channel}')
        return None

    bins = channel_bins(channel, concat([series for _, _, series in values]), options['minmax'])
    for device, location, series in values:
        colors = colors_of(series, bins)
        times = list(series.index)
        for index, date in enumerate(times[:-1]):
            if isnan(series.iloc[index]):
                continue
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[location.longitude, location.latitude]] * 2,
                },
                'properties': {
                    'times': [date.isoformat(), times[index + 1].isoformat()],
                    'popup': f'{device.id}: {round(series.iloc[index], 2)}',
                    'icon': options['icon'],
                    'iconstyle': {
                        'fillColor': str(colors.iloc[index]),
                        'fillOpacity': options['fillOpacity'],
                        'stroke': options['stroke'],
                        'radius': options['radius']
                    },
                    'style': {'weight': '0'},
                }
            })

    location = [sum(item[0] for item in locations) / len(locations), sum(item[1] for item in locations) / len(locations)] \
        if options['location'] == 'average' else options['location']
    m = Map(location=location, tiles=options['tiles'], zoom_start=options['zoom'])

    TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period=iso_period(options['period']),
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=options['max_speed'],
        loop_button=True,
        time_slider_drag_update=True,
        duration=iso_period(options['period'])
    ).add_to(m)

    if options['legend']:
        add_legend(m, legend_labels(bins))

    return m


def path_plot(self, channel=None, map_type='dynamic', devices='all',
              start_date=None, end_date=None, options=dict()):
    '''
    Creates a folium map with the path of devices with GPS (GPS_LAT and GPS_LONG columns), optionally
    colored by a channel
    Parameters
    -------
    channel: String
        None
        If None, shows the path, otherwise, the path colored by the channel
    map_type: String
        'dynamic'
        'dynamic' (with a time slider) or 'static'
    devices: list or 'all'
        Ids of the devices of a test to include
    start_date, end_date: String
        None
        Date convertible strings
    options: dict()
        Possible keys are (defaults in config._map_def_opt)
            location: list or 'average'
                Center of the map, or the average location of the path
            tiles: String
                Tiles for the folium.Map
            zoom: float
                Zoom to start with
            period: String
                Resample frequency of the path (pandas frequency, e.g. '1Min')
            minmax: bool
                Colors between the minimum and maximum of the data, instead of config._channel_bins
            markers: bool
                Circles on the points of a static map
            minimap, legend, radius, fillOpacity, stroke, stroke-width, icon, max_speed
    Returns
    -------
        folium.Map
    '''
    options = dict_fmerge(config._map_def_opt, options)
    options.setdefault('stroke-width', 3)

    mdev = map_devices(self, devices)
    if not mdev:
        logger.error('Requested devices not found')
        return None

    paths = []
    columns = ['GPS_LAT', 'GPS_LONG'] + ([channel] if channel is not None else [])
    for device in mdev:
        missing = [column for column in columns if column not in device.data.columns]
        if missing:
            # E.g. a fixed device in a test with mobile ones
            logger.warning(f'{missing} not in the data of device {device.id}, skipping')
            continue

        # Resample and fill the gaps of the path
        dfc = in_period(device.data[columns], start_date, end_date).resample(options['period']).mean()
        dfc = clean(dfc, 'fill')
        if len(dfc) < 2:
            logger.warning(f'Device {device.id} has no path in the period')
            continue
        paths.append((device, dfc))

    # The same bins for all devices: a color means the same values on every path
    labels = None
    if channel is not None and paths:
        bins = channel_bins(channel, concat([dfc[channel] for _, dfc in paths]), options['minmax'])
        labels = legend_labels(bins)

    features, points = [], []
    for device, dfc in paths:
        dfc['COLOR'] = colors_of(dfc[channel], bins) if channel is not None else config._map_colors_palette[0]

        times = list(dfc.index)
        for index, date in enumerate(times[:-1]):
            row, following = dfc.loc[date], dfc.loc[times[index + 1]]
            color = str(row['COLOR'])
            if color == 'nan' or isnan(row['GPS_LONG']) or isnan(row['GPS_LAT']) \
                    or isnan(following['GPS_LONG']) or isnan(following['GPS_LAT']):
                continue
            points.append([row['GPS_LAT'], row['GPS_LONG']])

            properties = {
                'icon': options['icon'],
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': options['fillOpacity'],
                    'stroke': options['stroke'],
                    'radius': options['radius']
                },
                'device': str(device.id),
                'timestamp': date.strftime('%Y-%m-%dT%H:%M:%S'),
                'coordinates': [following['GPS_LAT'], following['GPS_LONG']],
                'style': {
                    'color': color,
                    'stroke-width': options['stroke-width'],
                    'fillOpacity': options['fillOpacity']
                }
            }
            if channel is not None:
                properties['channel'] = channel
                properties['value'] = row[channel]
            if map_type == 'dynamic':
                properties['times'] = [date.isoformat(), times[index + 1].isoformat()]

            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[row['GPS_LONG'], row['GPS_LAT']], [following['GPS_LONG'], following['GPS_LAT']]],
                },
                'properties': properties
            })

    if not features:
        logger.error('No path to plot')
        return None
    featurecol = {'type': 'FeatureCollection', 'features': features}

    location = [sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)] \
        if options['location'] == 'average' else options['location']
    m = Map(location=location, tiles=options['tiles'], zoom_start=options['zoom'])

    if map_type == 'static':
        if options['markers']:
            for feature in features:
                Circle(
                    location=[feature['geometry']['coordinates'][0][1], feature['geometry']['coordinates'][0][0]],
                    fill=True,
                    radius=feature['properties']['iconstyle']['radius'],
                    color=feature['properties']['iconstyle']['fillColor'],
                    fill_opacity=feature['properties']['iconstyle']['fillOpacity']
                ).add_to(m)

        if channel is not None:
            fields = ['device', 'channel', 'timestamp', 'coordinates', 'value']
            aliases = ['Device:', 'Sensor:', 'Timestamp:', 'Coordinates:', 'Reading:']
        else:
            fields = ['device', 'timestamp', 'coordinates']
            aliases = ['Device:', 'Timestamp:', 'Coordinates:']

        GeoJson(featurecol,
            tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=True, labels=True,
                                   max_width=800),
            popup=GeoJsonPopup(fields=fields, aliases=aliases, localize=True, labels=True, max_width=800),
            style_function=lambda x: {
                'color': x['properties']['style']['color'],
                'weight': x['properties']['style']['stroke-width'],
                'fillOpacity': x['properties']['style']['fillOpacity']},
        ).add_to(m)

    elif map_type == 'dynamic':
        TimestampedGeoJson(
            featurecol,
            period=iso_period(options['period']),
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=options['max_speed'],
            loop_button=True,
            time_slider_drag_update=True
        ).add_to(m)

    else:
        logger.error(f'Not supported map type {map_type}')
        return None

    if options['minimap']:
        MiniMap(toggle_display=True, tile_layer=options['tiles']).add_to(m)

    if options['legend'] and labels is not None:
        add_legend(m, labels)

    return m
