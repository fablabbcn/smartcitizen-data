''' Data of the device plots, with missing values filled or dropped '''
from types import SimpleNamespace

import numpy as np
import pytest
from pandas import DataFrame

pytest.importorskip('matplotlib')
from scdata.plot.tools import prepare_device_data


def plot_data(index, clean_na, frequency=None):
    device = SimpleNamespace(id=1, data=DataFrame({'TEMP': [1.0, np.nan, 3.0] + [4.0] * 7}, index=index))
    df, subplots, _ = prepare_device_data(device, {'1': {'channel': 'TEMP', 'subplot': 1}},
                                          {'frequency': frequency, 'clean_na': clean_na, 'min_date': None,
                                           'max_date': None})
    return df, subplots


def test_fill_uses_the_previous_value(index):
    df, subplots = plot_data(index, 'fill')

    assert df['TEMP'].tolist()[:3] == [1.0, 1.0, 3.0]
    assert subplots == [['TEMP']]


def test_drop_removes_the_missing_values(index):
    df, _ = plot_data(index, 'drop')

    assert len(df) == 9 and df['TEMP'].notna().all()


def test_resample_with_the_units_before_pandas_3(index):
    df, _ = plot_data(index, None, frequency='5T')

    # The missing reading is not counted in the mean
    assert df['TEMP'].tolist() == [3.0, 4.0]
