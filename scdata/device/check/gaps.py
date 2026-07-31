from scdata._config import config
from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult
from pandas import Timedelta, DataFrame

import pandas as pd

# def find_gap_in_column(dataframe, frequency):
# Attempt to avoid having to resample -> Works, but to determine gap size only

    # df = dataframe.copy()
    # df = df.dropna()
    # df['time'] = df.index
    # df['time_lag'] = df['time'].shift(1)
    # df['time_delta'] = df['time'] - df['time_lag']
    # df['gap'] = df['time_delta'] > Timedelta(minutes=gap_size)
    # df['gap_size'] = df.loc[df['gap'], 'time_delta']

    # return df['gap']

# def find_gap_in_column(
#     s: pd.Series,
#     frequency: int | None = None,
#     gap_size: int = 5,
#     jitter_tolerance_sec: int = 5
# ):
#     """
#     Detect gaps in timeseries:
#     - frequency: expected frequency in minutes
#     - gap_size: minimum gap size in minutes
#     - jitter_tolerance_sec: tolerance to jitter in seconds
#     """

#     s = s.sort_index()
#     s = s[~s.index.duplicated()]

#     if len(s) < 2:
#         return pd.Series(False, index=s.index)

#     dt = s.index.to_series().diff().dropna()
#     freq_sec = frequency * 60
#     gap_threshold = freq_sec + jitter_tolerance_sec

#     gap_events = dt > pd.Timedelta(seconds=gap_threshold)
#     gap_mask = pd.Series(False, index=s.index)

#     for idx in dt.index[gap_events]:
#         start = idx - dt.loc[idx]
#         end = idx

#         gap_duration_min = dt.loc[idx].total_seconds() / 60

#         if gap_duration_min >= gap_size:
#             gap_mask.loc[start:end] = True

#     return gap_mask

def find_gaps(dataframe, **kwargs):

    # default_gap_size_minutes = kwargs.get('default_gap_size_minutes',
    #     config._default_gap_size_minutes)
    # gap_sizes = kwargs.get('gap_sizes', None)

    # default_frequency_minutes = kwargs.get('default_frequency_minutes', 1)
    # frequencies = kwargs.get('frequencies', None)

    # gaps = []
    df = dataframe.copy()

    cols = []
    for col in df.columns:
        # gap_size = default_gap_size_minutes
        # frequency = default_frequency_minutes
        if '__' in col: continue # Internal code for healthchecks

        # if gap_sizes is not None:
        #     for gap_size_group in gap_sizes:
        #         if col in gap_size_group["columns"]:
        #             gap_size = gap_size_group["gap_size_minutes"]
        #             break

        # if frequencies is not None:
        #     for frequency_group in frequencies:
        #         if col in frequency_group["columns"]:
        #             frequency = frequency_group["frequency_minutes"]
        #             break

        # logger.info (f'Calculating gaps for {col}, using {frequency} minutes. Gap size: {gap_size} minutes')
        logger.info (f'Calculating gaps for {col}')
        # df[f'__{col}'] = find_gap_in_column(df[col], frequency, gap_size)
        df[f'__{col}'] = df.loc[:, col].isna()
        cols.append(f'__{col}')

    return ProcessResult(df[cols], StatusCode.SUCCESS)