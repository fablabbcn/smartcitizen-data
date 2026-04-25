from scdata._config import config
from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult

def find_implausible_values(dataframe, **kwargs):

    implausible_values = kwargs.get('implausible_values', config._default_implausible_values)

    df = dataframe.copy()
    cols = []

    for item in implausible_values:
        col = item['column']
        min_val = item['limits'][0]
        max_val = item['limits'][1]
        if col not in df.columns:
            logger.warning(f'{col} not in columns. Skipping')
            continue

        logger.info (f'Calculating implausible values for {col} using [{min_val}, {max_val}]')
        df[f'__{col}'] = (df.loc[:, col] < min_val) | (df.loc[:, col] > max_val)

        cols.append(f'__{col}')

    return ProcessResult(df[cols], StatusCode.SUCCESS)