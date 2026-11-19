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


def test_device_metric_map_without_data():
    test = SimpleNamespace(devices=[device(1, None, None, readings({'NOISE_A': np.ones(48)}))])

    assert device_metric_map(test, 'NOISE_A') is None


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
