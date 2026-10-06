from pandas import DatetimeIndex, Timedelta

from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult

def find_flat_values(dataframe, **kwargs):
    '''
    Flags values where the rolling standard deviation is below limit_rolling_std

    Parameters
    ----------
        flat_window_minutes: int
            1000
            Window in minutes. Requires a DatetimeIndex. Values are only flagged
            once a full window of data is available
        flat_sensor_window: int
            None
            Window in rows. Used instead of flat_window_minutes if set
        limit_rolling_std: float
            1e-5
            Standard deviation below which values are flat
        columns: list
            All columns
            Columns to check
    '''

    flat_window_minutes = kwargs.get('flat_window_minutes', 1000)
    flat_sensor_window = kwargs.get('flat_sensor_window', None)
    limit_rolling_std = kwargs.get('limit_rolling_std', 1e-5)
    columns = kwargs.get('columns', list(dataframe.columns))

    df = dataframe.copy()
    cols = []

    if flat_sensor_window is not None:
        window = flat_sensor_window
        full_window = None
        logger.info(f'Flat window size: {flat_sensor_window} rows. STD Limit: {limit_rolling_std}')
    elif isinstance(df.index, DatetimeIndex):
        window = f'{flat_window_minutes}min'
        # Time based windows are computed with partial data at the start
        full_window = df.index >= df.index.min() + Timedelta(minutes=flat_window_minutes)
        logger.info(f'Flat window size: {flat_window_minutes} minutes. STD Limit: {limit_rolling_std}')
    else:
        logger.error('flat_window_minutes requires a DatetimeIndex')
        return ProcessResult(None, StatusCode.ERROR_WRONG_INDEX)

    for col in columns:
        if '__' in col: continue # Internal code for healthchecks
        if col not in df.columns:
            logger.warning(f'{col} not in columns. Skipping')
            continue

        logger.info (f'Calculating flat values for {col}')
        df[f'__{col}'] = df[col].rolling(window=window).std() < limit_rolling_std
        if full_window is not None:
            df[f'__{col}'] &= full_window

        cols.append(f'__{col}')

    return ProcessResult(df[cols], StatusCode.SUCCESS)
