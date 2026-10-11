from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

folium = pytest.importorskip('folium')

from scdata.plot.maps import device_metric_map, iso_period, path_plot


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
