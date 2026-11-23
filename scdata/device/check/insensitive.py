import numpy as np
from pandas import Grouper

from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult

def find_insensitive_electrodes(dataframe, **kwargs):
    '''
    Flags the periods in which the working electrode (WE) of an electrochemical sensor barely changes beyond what
    its auxiliary electrode (AE) explains: the sensor does not respond to the gas, and the WE follows the AE and the
    temperature. In each window, WE is fitted on AE (averages of 10 minutes), and the range (95th - 5th percentile)
    of what is left is compared with min_range_mv

    Parameters
    ----------
        electrodes: list
            [{'channel': 'SO2', 'we': 'SO2_WE', 'ae': 'SO2_AE'}]
            Electrode pairs, in V. Flags are named after the channel
        min_range_mv: float
            3
            Range (mV) of the WE not explained by the AE below which the window is flagged
        window_hours: int
            72
            Length of the windows. Shorter windows flag sensors that are working, in clean air
        min_coverage: float
            0.5
            Share of the 10 minute averages of a window that need values for it to be checked
    '''

    electrodes = kwargs.get('electrodes', [])
    min_range_mv = kwargs.get('min_range_mv', 3)
    window_hours = kwargs.get('window_hours', 72)
    min_coverage = kwargs.get('min_coverage', 0.5)

    if not electrodes:
        logger.error('No electrodes to check')
        return ProcessResult(None, StatusCode.ERROR_MISSING_INPUTS)

    df = dataframe.copy()
    cols = []

    for electrode in electrodes:
        channel, we, ae = electrode.get('channel'), electrode.get('we'), electrode.get('ae')
        if we not in df.columns or ae not in df.columns:
            logger.info(f'{we} or {ae} not in columns. Skipping')
            continue

        flag = f'__{channel or we}'
        df[flag] = False
        averages = df[[we, ae]].resample('10min').mean() * 1000 # in mV
        for start, window in averages.groupby(Grouper(freq=f'{window_hours}h')):
            window = window.dropna()
            if len(window) < max(min_coverage * window_hours * 6, 2):
                continue
            if window[ae].nunique() > 1:
                a, b = np.polyfit(window[ae], window[we], 1)
                residual = window[we] - (a * window[ae] + b)
            else:
                # A constant AE (0 if disconnected) cannot be fitted: WE around its mean
                residual = window[we] - window[we].mean()
            if residual.quantile(0.95) - residual.quantile(0.05) < min_range_mv:
                df.loc[(df.index >= start) & (df.index < start + np.timedelta64(window_hours, 'h')), flag] = True
        # Only the rows with readings of the electrodes, and of the channel if it is there (summaries count its values)
        df[flag] &= df[we].notna() & df[ae].notna()
        if channel in df.columns:
            df[flag] &= df[channel].notna()

        cols.append(flag)

    return ProcessResult(df[cols], StatusCode.SUCCESS)
