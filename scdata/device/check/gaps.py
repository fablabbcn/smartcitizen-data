from pandas import DatetimeIndex, Series, Timedelta

from scdata._config import config
from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult

# Readings arrive with some jitter around their frequency
JITTER = Timedelta(seconds=30)


def gap_intervals(series, gap_size_minutes, frequency_minutes, start=None, end=None):
    '''
    Periods without readings of the series longer than gap_size_minutes (and than the frequency
    of the sensor), as [(start, end)]. The start and end of the period checked (by default, of
    the series) count as readings, so missing data at either end is a gap too
    '''
    start = series.index.min() if start is None else start
    end = series.index.max() if end is None else end
    threshold = max(Timedelta(minutes=gap_size_minutes), Timedelta(minutes=frequency_minutes)) + JITTER

    bounds = [start] + list(series.dropna().index) + [end]
    return [(before, after) for before, after in zip(bounds[:-1], bounds[1:]) if after - before > threshold]


def find_gap_in_column(series, gap_size_minutes, frequency_minutes, start=None, end=None):
    ''' Rows of the series without reading that fall in a gap (see gap_intervals) '''
    index = series.index
    gap = Series(False, index=index)
    for before, after in gap_intervals(series, gap_size_minutes, frequency_minutes, start, end):
        gap |= (index >= before) & (index <= after) & series.isna()
    return gap


def column_setting(groups, column, key, default):
    ''' Value of key in the first group that lists the column ({"columns": [...], key: value}) '''
    for group in groups or []:
        if column in group.get('columns', []):
            return group[key]
    return default


def find_gaps(dataframe, **kwargs):
    '''
    Flags the rows of each column that fall in a gap without readings

    Parameters
    ----------
        default_gap_size_minutes: int
            5
            Shortest period without readings that is a gap
        default_frequency_minutes: int
            1
            Expected time between readings. Periods up to this long are never gaps
        gap_sizes: list
            None
            Gap sizes of some columns: [{"columns": [...], "gap_size_minutes": 10}]
        frequencies: list
            None
            Frequencies of some columns: [{"columns": [...], "frequency_minutes": 5}]
        columns: list
            All columns
            Columns to check
    '''
    if not isinstance(dataframe.index, DatetimeIndex):
        logger.error('find_gaps requires a DatetimeIndex')
        return ProcessResult(None, StatusCode.ERROR_WRONG_INDEX)

    default_gap_size = kwargs.get('default_gap_size_minutes', config._default_gap_size_minutes)
    default_frequency = kwargs.get('default_frequency_minutes', 1)
    gap_sizes = kwargs.get('gap_sizes')
    frequencies = kwargs.get('frequencies')
    columns = kwargs.get('columns', list(dataframe.columns))

    df = dataframe.sort_index()
    result = df[[]].copy()
    intervals = dict()
    for col in columns:
        if '__' in col: continue # Internal code for healthchecks
        if col not in df.columns:
            logger.warning(f'{col} not in columns. Skipping')
            continue

        gap_size = column_setting(gap_sizes, col, 'gap_size_minutes', default_gap_size)
        frequency = column_setting(frequencies, col, 'frequency_minutes', default_frequency)
        logger.info(f'Calculating gaps for {col}, every {frequency} minutes. Gap size: {gap_size} minutes')
        result[f'__{col}'] = find_gap_in_column(df[col], gap_size, frequency)
        # Rows only exist when some sensor sent a reading: the periods show gaps without any row
        intervals[f'__{col}'] = gap_intervals(df[col], gap_size, frequency)

    return ProcessResult(result, StatusCode.SUCCESS, intervals=intervals)
