import pytest
from numpy import isnan, nan
from pandas import DataFrame

from scdata.device.process.error_codes import StatusCode
from scdata.device.process.timeseries import baseline_als, clean_ts, poly_ts, rolling_avg, time_derivative


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


@pytest.fixture
def squares():
    # t² sampled every 2 minutes, with t in minutes
    import pandas as pd
    index = pd.date_range('2026-01-01', periods=20, freq='2min', tz='UTC')
    minutes = [2.0 * i for i in range(20)]
    return DataFrame({'A': [t ** 2 for t in minutes]}, index=index), minutes


def test_time_derivative_with_the_next_reading(squares):
    df, minutes = squares
    result = time_derivative(df, name='A', gaussian_filter1d=False)

    assert result.status_code == StatusCode.SUCCESS
    # ((t + 2)² - t²) / 2 minutes
    assert list(result.data[:-1]) == [2 * t + 2 for t in minutes[:-1]]
    assert result.data.iloc[-1] == 0


def test_time_derivative_smoothed(squares):
    df, minutes = squares
    result = time_derivative(df, name='A')

    assert result.status_code == StatusCode.SUCCESS
    assert len(result.data) == len(df)
    # 2t away from the edges, within the error of the truncated gaussian kernel
    assert list(result.data[5:-5]) == pytest.approx([2 * t for t in minutes[5:-5]], rel=1e-4)


@pytest.mark.parametrize('kwargs, status', [({}, StatusCode.ERROR_MISSING_INPUTS),
                                            ({'name': 'X'}, StatusCode.ERROR_MISSING_CHANNEL)])
def test_time_derivative_errors(squares, kwargs, status):
    assert time_derivative(squares[0], **kwargs).status_code == status


@pytest.mark.parametrize('intervals', [[1, 3] * 10, [1] * 10 + [15] + [1] * 9, [1.1, 0.9, 1.05] * 7])
def test_time_derivative_smoothed_with_irregular_intervals(intervals):
    import pandas as pd
    minutes = [0.0]
    for interval in intervals:
        minutes.append(minutes[-1] + interval)
    index = pd.Timestamp('2026-01-01', tz='UTC') + pd.to_timedelta(minutes, unit='min')
    result = time_derivative(DataFrame({'A': [3 * t + 1 for t in minutes]}, index=index), name='A')

    assert result.status_code == StatusCode.SUCCESS
    # The slope of a linear signal does not depend on the intervals, including at the edges
    assert list(result.data) == pytest.approx([3] * len(minutes))


def test_time_derivative_smoothed_with_missing_readings():
    import pandas as pd
    index = pd.date_range('2026-01-01', periods=20, freq='1min', tz='UTC')
    values = [3.0 * t for t in range(20)]
    values[5] = values[6] = nan
    result = time_derivative(DataFrame({'A': values}, index=index), name='A')

    assert result.status_code == StatusCode.SUCCESS
    # Missing readings remain missing and do not affect the neighbouring derivatives
    assert result.data.isna().tolist() == [i in (5, 6) for i in range(20)]
    assert list(result.data.dropna()) == pytest.approx([3] * 18)


def test_time_derivative_smoothed_with_an_isolated_reading():
    import pandas as pd
    index = pd.DatetimeIndex(['2026-01-01 00:00', '2026-01-01 00:01', '2026-01-01 00:02', '2026-01-05 00:00'], tz='UTC')
    result = time_derivative(DataFrame({'A': [0.0, 2.0, 4.0, 10.0]}, index=index), name='A')

    assert result.status_code == StatusCode.SUCCESS
    assert list(result.data[:3]) == pytest.approx([2, 2, 2])
    assert isnan(result.data.iloc[-1])


@pytest.mark.parametrize('gaussian_filter1d', [True, False])
@pytest.mark.parametrize('minutes', [[0, 1, 1, 2], [0, 2, 1, 3, 4]])
def test_time_derivative_with_timestamps_that_do_not_increase(minutes, gaussian_filter1d):
    import pandas as pd
    index = pd.Timestamp('2026-01-01', tz='UTC') + pd.to_timedelta(minutes, unit='min')
    result = time_derivative(DataFrame({'A': [float(m) for m in minutes]}, index=index), name='A',
                             gaussian_filter1d=gaussian_filter1d)

    assert result.status_code == StatusCode.ERROR_WRONG_INDEX
