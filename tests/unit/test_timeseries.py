import pytest
from numpy import nan
from pandas import DataFrame

from scdata.device.process.error_codes import StatusCode
from scdata.device.process.timeseries import clean_ts, poly_ts, rolling_avg


@pytest.fixture
def df():
    return DataFrame({'A': [1.0, 2.0, 3.0, 4.0], 'B': [10.0, 20.0, 30.0, 40.0], 'C': [100.0, 5.0, 200.0, 50000.0]})


def test_poly_ts(df):
    result = poly_ts(df, channels=['A', 'B'], coefficients=[2, -1], extra_term=5)

    assert result.status_code == StatusCode.SUCCESS
    assert list(result.data) == [-3.0, -11.0, -19.0, -27.0]


def test_poly_ts_exponents(df):
    result = poly_ts(df, channels=['A'], exponents=[2])

    assert list(result.data) == [1.0, 4.0, 9.0, 16.0]


@pytest.mark.parametrize('kwargs, status', [
    ({}, StatusCode.ERROR_MISSING_INPUTS),
    ({'channels': ['A', 'MISSING']}, StatusCode.ERROR_MISSING_CHANNEL),
])
def test_poly_ts_errors(df, kwargs, status):
    assert poly_ts(df, **kwargs).status_code == status


def test_clean_ts_limits(df):
    result = clean_ts(df, name='C', limits=[10, 1000], window=1)

    assert result.status_code == StatusCode.SUCCESS
    assert result.data.tolist()[0] == 100.0
    assert result.data.isna().tolist() == [False, True, False, True]


@pytest.mark.parametrize('kwargs, status', [
    ({}, StatusCode.ERROR_MISSING_INPUTS),
    ({'name': 'MISSING'}, StatusCode.ERROR_MISSING_CHANNEL),
])
def test_clean_ts_errors(df, kwargs, status):
    assert clean_ts(df, **kwargs).status_code == status


def test_rolling_avg(df):
    result = rolling_avg(df, name='A', window_size=2)

    assert result.status_code == StatusCode.SUCCESS
    assert result.data.tolist()[1:] == [1.5, 2.5, 3.5]


@pytest.mark.parametrize('kwargs, status', [
    ({}, StatusCode.ERROR_MISSING_INPUTS),
    ({'name': 'MISSING'}, StatusCode.ERROR_MISSING_CHANNEL),
])
def test_rolling_avg_errors(df, kwargs, status):
    assert rolling_avg(df, **kwargs).status_code == status
