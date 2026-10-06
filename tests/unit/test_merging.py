import pandas as pd
import pytest

from scdata.tools.merging import compare_dataframes, dataframe_row_diff


def frame(values, column='A', dtype=None, start='2026-01-01', tz='UTC'):
    index = pd.date_range(start, periods=len(values), freq='1min', tz=tz)
    return pd.DataFrame({column: pd.array(values, dtype=dtype)}, index=index)


def test_row_diff_cutoff_with_empty_first_frame():
    df2 = frame([1.0, 2.0, 3.0])

    only_1, only_2 = dataframe_row_diff(df2.iloc[0:0], df2, cutoff='2026-01-01 00:01:00')

    assert only_1.empty
    assert len(only_2) == 2


def test_row_diff_cutoff_normalised_per_frame():
    df1 = frame([1.0, 2.0, 3.0], tz=None)
    df2 = frame([1.0, 2.0, 3.0, 4.0])

    only_1, only_2 = dataframe_row_diff(df1, df2, cutoff=pd.Timestamp('2026-01-01 00:02:00', tz='UTC'))

    assert len(only_1) == 1
    assert len(only_2) == 2


def test_compare_equal_frames():
    report = compare_dataframes(frame([1.0, None]), frame([1.0, None]))

    assert report['num_value_differences'] == 0


def test_compare_counts_nullable_differences():
    report = compare_dataframes(frame([1, None, 3], dtype='Int64'), frame([1, 2, 4], dtype='Int64'))

    assert report['num_value_differences'] == 2
    assert report['value_differences_per_column'] == {'A': 2}


def test_compare_rejects_duplicated_labels():
    df1 = frame([1.0, 2.0])
    df1.index = [df1.index[0], df1.index[0]]

    with pytest.raises(ValueError):
        compare_dataframes(df1, frame([1.0]))
