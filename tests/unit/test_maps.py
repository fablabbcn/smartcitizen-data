import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

folium = pytest.importorskip('folium')

from scdata._config import config
from scdata.plot.maps import channel_bins, colors_of, device_metric_map, iso_period, path_plot


def device(device_id, latitude, longitude, data):
    location = SimpleNamespace(latitude=latitude, longitude=longitude)
    return SimpleNamespace(id=device_id, data=data, handler=SimpleNamespace(json=SimpleNamespace(location=location)))


def readings(columns, periods=48, freq='1h'):
    index = pd.date_range('2026-10-01', periods=periods, freq=freq, tz='Europe/Madrid')
    return pd.DataFrame({column: values for column, values in columns.items()}, index=index)


def test_iso_period():
    assert iso_period('10Min') == 'P0DT0H10M0S'
    assert iso_period('1W') == 'P7DT0H0M0S'


def test_device_metric_map_of_a_test():
    test = SimpleNamespace(devices=[
        device(1, 41.4, 2.18, readings({'NOISE_A': np.linspace(40, 80, 48)})),
        device(2, 41.5, 2.2, readings({'NOISE_A': np.linspace(50, 60, 48)})),
        # No location, no channel: skipped
        device(3, None, None, readings({'NOISE_A': np.ones(48)})),
        device(4, 41.6, 2.3, readings({'TEMP': np.ones(48)})),
    ])

    m = device_metric_map(test, 'NOISE_A', start_date='2026-10-01', end_date='2026-10-02 12:00',
                          options={'period': '6h', 'location': 'average'})

    html = m.get_root().render()
    assert isinstance(m, folium.Map) and 'timeDimension' in html
    assert m.location == pytest.approx([41.45, 2.19])
    # Devices 1 and 2, 6 periods of 6 hours in the 36 hours, the last one has no next time
    assert html.count('"popup": "1:') == 5 and html.count('"popup": "2:') == 5
    assert 'Legend' in html


def test_device_metric_map_without_data():
    test = SimpleNamespace(devices=[device(1, None, None, readings({'NOISE_A': np.ones(48)}))])

    assert device_metric_map(test, 'NOISE_A') is None


def test_path_plot():
    data = readings({'GPS_LAT': np.linspace(41.38, 41.40, 60), 'GPS_LONG': np.linspace(2.17, 2.19, 60),
                     'PM_25': np.linspace(5, 120, 60)}, periods=60, freq='1min')
    test = SimpleNamespace(devices=[device(7, 41.4, 2.18, data)])

    dynamic = path_plot(test, channel='PM_25', options={'period': '5Min', 'location': 'average', 'zoom': 14})
    static = path_plot(test, channel='PM_25', map_type='static', devices=[7], options={'period': '5Min'})

    assert isinstance(dynamic, folium.Map) and isinstance(static, folium.Map)
    assert dynamic.location == pytest.approx([41.389, 2.179], abs=1e-3)
    assert static.get_root().render().count('"device": "7"') == 11


def test_path_plot_needs_gps():
    test = SimpleNamespace(devices=[device(7, 41.4, 2.18, readings({'PM_25': np.ones(48)}))])

    assert path_plot(test, channel='PM_25') is None


def test_colors_of_include_the_edges():
    palette = config._map_colors_palette
    bins = np.linspace(0, 10 * len(palette), len(palette) + 1)
    colors = colors_of(pd.Series([0, bins[-1], 1000, -5, np.nan]), bins)

    # The minimum, values above or below the bins with the closest color, missing values without
    assert list(colors[:4]) == [palette[0], palette[-1], palette[-1], palette[0]]
    assert pd.isna(colors.iloc[4])


def test_bins_of_a_constant_series():
    bins = channel_bins('X', pd.Series([5.0, 5.0]), minmax=True)

    assert bins[0] < 5 < bins[-1]
    assert not pd.isna(colors_of(pd.Series([5.0, 5.0]), bins)).any()


def path(start, values):
    return readings({'GPS_LAT': np.linspace(41.38, 41.40, 60), 'GPS_LONG': np.linspace(start, start + 0.02, 60),
                     'PM_25': values}, periods=60, freq='1min')


def test_path_plot_keeps_the_lowest_segment_with_minmax():
    test = SimpleNamespace(devices=[device(7, 41.4, 2.18, path(2.17, np.linspace(5, 120, 60)))])

    m = path_plot(test, channel='PM_25', map_type='static', options={'period': '5Min', 'minmax': True})

    assert m.get_root().render().count('"device": "7"') == 11


def test_path_plot_skips_devices_without_gps_and_shares_the_bins():
    test = SimpleNamespace(devices=[
        device(7, 41.4, 2.18, path(2.17, np.linspace(5, 20, 60))),
        device(8, 41.4, 2.18, path(2.27, np.linspace(100, 120, 60))),
        # A fixed device: no GPS columns
        device(9, 41.4, 2.18, readings({'PM_25': np.ones(48)})),
    ])

    m = path_plot(test, channel='PM_25', map_type='static', options={'period': '5Min', 'minmax': True})

    html = m.get_root().render()
    assert html.count('"device": "7"') == 11 and html.count('"device": "8"') == 11
    assert 'Legend' in html
    palette = config._map_colors_palette
    # Device 7 is at the bottom of the shared range, device 8 at the top (with bins per device, both span all colors)
    colors = {}
    for device_id, color in re.findall(r'"device": "(\d)".*?"color": "([^"]+)"', html):
        colors.setdefault(device_id, set()).add(color)
    assert palette[0] in colors['7'] and palette[-1] not in colors['7']
    assert palette[-1] in colors['8'] and palette[0] not in colors['8']
