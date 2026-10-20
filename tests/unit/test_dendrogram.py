''' Dendrogram of the channels of the devices of a test, or of a device '''
from types import SimpleNamespace

import numpy as np
import pytest
from pandas import DataFrame, date_range

pytest.importorskip('matplotlib')
pytest.importorskip('seaborn')
import matplotlib
matplotlib.use('Agg')
from scdata.plot.ts_dendrogram import ts_dendrogram


def device(id, shift):
    index = date_range('2026-01-01', periods=60, freq='1min', tz='UTC')
    values = np.sin(np.arange(60) / 10 + shift)
    data = DataFrame({'TEMP': values, 'HUM': -values, 'NAME': 'x'}, index=index)
    data.iloc[5, 0] = np.nan
    return SimpleNamespace(id=id, data=data)


def test_devices_of_a_test():
    test = SimpleNamespace(devices=[device(1, 0), device(2, 0.1)])

    Z = ts_dendrogram(test, channels=['TEMP', 'HUM'], options={'show': False, 'clean_na': 'fill', 'frequency': '5T'})

    # Four channels result in three merges
    assert Z.shape == (3, 4)
    # The TEMP channels of both devices are merged first, as are the HUM channels
    assert sorted(Z[:2, :2].astype(int).tolist()) == [[0, 2], [1, 3]]


def test_a_device():
    Z = ts_dendrogram(device(1, 0), options={'show': False, 'clean_na': 'drop'})

    # Only the numeric channels (TEMP and HUM) are used
    assert Z.shape == (1, 4)


@pytest.mark.parametrize('clean_na', [None, 'fill'])
def test_devices_with_different_timestamps(clean_na):
    late = device(2, 0.1)
    # The second device starts ten minutes later
    late.data.iloc[:10, :2] = np.nan
    test = SimpleNamespace(devices=[device(1, 0), late])

    Z = ts_dendrogram(test, channels=['TEMP', 'HUM'], options={'show': False, 'clean_na': clean_na})

    assert Z.shape == (3, 4)
    assert np.isfinite(Z).all()
