from scdata.tools.custom_logger import logger

from pandas import Series, Timedelta
import numpy as np

def infer_sampling_rate(series: Series) -> int | None:
    '''Infer the sampling rate of the given timeseries, rounded to the
    closest minute.
    '''

    time_differences = series.index.diff().value_counts()
    most_common = time_differences.index[0]

    minutes = most_common / Timedelta("1min")
    integer_minutes = round(minutes)

    if abs(integer_minutes - minutes) > 0.05:
        logger.warning('Rounded a time difference with more than 5% error')
        return None

    return integer_minutes


def mode_ratio(series: Series, ignore_zeroes=True) -> int:
    '''Count the percentage of times the most common value appears in the series,
    ignoring zeroes and NaNs.'''

    if ignore_zeroes:
        # Replace zeroes with random so that they don't impact value count
        series = series.where(series!=0.0, np.random.random(size=series.size))

    mode_count = series.value_counts().iloc[0]

    return mode_count / series.count()


def count_nas(series: Series) -> int:
    '''Count the number of NaN values in the series.'''
    return series.isna().sum()