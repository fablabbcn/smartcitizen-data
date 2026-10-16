import pytest
from numpy import nan
from pandas import DataFrame

from scdata.device.process.error_codes import StatusCode
from scdata.device.process.timeseries import baseline_als, clean_ts, poly_ts, rolling_avg


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


@pytest.mark.parametrize('values', [[nan, nan, 400.0], [nan, nan, nan]])
def test_baseline_als_with_few_values(values):
    result = baseline_als(DataFrame({'A': values}), name='A')

    assert result.status_code == StatusCode.SUCCESS
    assert result.data.tolist() == [value for value in values if value == value]


@pytest.mark.parametrize('dates', [
    ['2026-10-16T19:57:35Z', '2026-10-16T19:58:35.123Z'],
    ['2026-10-16 19:57:35+00:00', '2026-10-16 19:58:35+00:00'],
])
def test_localise_iso_dates_without_warnings(dates):
    import warnings
    from pandas import Index, Timestamp
    from scdata.tools.date import localise_date

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = localise_date(Index(dates), 'Europe/Madrid')

    assert result[0] == Timestamp(dates[0]) and str(result.tz) == 'Europe/Madrid'


def test_localise_with_a_format_and_other_dates():
    from pandas import Timestamp
    from scdata.tools.date import localise_date

    assert localise_date('02-01-2024', 'UTC', dateformat='%d-%m-%Y') == Timestamp('2024-01-02', tz='UTC')
    assert localise_date('01/02/2024', 'UTC') == Timestamp('2024-01-02', tz='UTC')
