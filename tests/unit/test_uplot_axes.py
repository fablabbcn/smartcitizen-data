''' Side of the y axis of ts_uplot '''
import re

import pandas as pd
import pytest

pytest.importorskip('jinja2')
pytest.importorskip('IPython')
# Only a method of Device and Test under IPython
from scdata.plot.ts_uplot import ts_uplot


@pytest.fixture
def device(make_device):
    device = make_device({'channels': []}, [])
    device.data = pd.DataFrame({'TEMP': [20.0, 21.0], 'HUM': [50.0, 51.0]},
                               index=pd.date_range('2026-01-01', periods=2, freq='1h', tz='UTC'))
    device.loaded = True
    return device


def y_sides(device, traces):
    html = ts_uplot(device, traces=traces, options={'html': True})
    return [int(side) for side in re.findall(r"'side': (\d)", html)]


def test_the_y_axis_is_on_the_left_by_default(device):
    assert y_sides(device, {'1': {'channel': 'TEMP', 'subplot': 1}}) == [3]


def test_the_y_axis_is_on_the_right_if_all_traces_are(device):
    assert y_sides(device, {'1': {'channel': 'TEMP', 'subplot': 1, 'side': 'right'}}) == [1]


def test_traces_on_both_sides_have_two_axes(device):
    traces = {'1': {'channel': 'TEMP', 'subplot': 1, 'side': 'left'},
              '2': {'channel': 'HUM', 'subplot': 1, 'side': 'right'}}
    assert y_sides(device, traces) == [3, 1]
