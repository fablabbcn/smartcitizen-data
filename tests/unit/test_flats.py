import numpy as np
import pandas as pd
import pytest

from scdata.device.check.flats import find_flat_values
from scdata.device.process.error_codes import StatusCode


@pytest.fixture
def df():
    # 60 changing values, then 60 constant values, every minute
    index = pd.date_range('2026-01-01', periods=120, freq='1min', tz='UTC')
    return pd.DataFrame({'A': np.r_[np.arange(60.0), np.full(60, 5.0)], 'B': np.arange(120.0)}, index=index)


def test_flat_window_minutes(df):
    result = find_flat_values(df, flat_window_minutes=30, columns=['A', 'B'])

    assert result.status_code == StatusCode.SUCCESS
    # The 30 minute window only holds constant values from minute 89
    assert result.data['__A'].sum() == 31
    assert result.data['__A'].idxmax() == df.index[89]
    assert not result.data['__B'].any()


def test_flat_window_minutes_needs_full_window(df):
    df['A'] = 5.0

    result = find_flat_values(df, flat_window_minutes=30, columns=['A'])

    assert result.data['__A'].idxmax() == df.index[30]


def test_flat_window_minutes_depends_on_time(df):
    # Same rows sampled every 2 minutes: the window (t - 30min, t] holds 15 rows,
    # so it only holds constant values from row 74
    df.index = pd.date_range('2026-01-01', periods=120, freq='2min', tz='UTC')

    result = find_flat_values(df, flat_window_minutes=30, columns=['A'])

    assert result.data['__A'].idxmax() == df.index[74]


def test_flat_sensor_window_in_rows(df):
    result = find_flat_values(df, flat_sensor_window=30, columns=['A'])

    assert result.data['__A'].sum() == 31


def test_flat_window_minutes_requires_datetime_index(df):
    result = find_flat_values(df.reset_index(drop=True), flat_window_minutes=30)

    assert result.status_code == StatusCode.ERROR_WRONG_INDEX
