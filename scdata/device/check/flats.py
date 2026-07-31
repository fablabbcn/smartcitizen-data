from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult

def find_flat_values(dataframe, **kwargs):

    flat_sensor_window = kwargs.get('flat_sensor_window', 1000)
    limit_rolling_std = kwargs.get('limit_rolling_std', 1e-5)
    columns = kwargs.get('columns', list(dataframe.columns))

    df = dataframe.copy()
    cols = []

    logger.info(f'Flat window size: {flat_sensor_window} minutes. STD Limit: {limit_rolling_std}')
    for col in columns:
        if '__' in col: continue # Internal code for healthchecks
        if col not in df.columns:
            logger.warning(f'{col} not in columns. Skipping')
            continue

        logger.info (f'Calculating flat values for {col}')
        df[f'__{col}'] = df[col].rolling(window=flat_sensor_window).std() < limit_rolling_std

        cols.append(f'__{col}')

    return ProcessResult(df[cols], StatusCode.SUCCESS)