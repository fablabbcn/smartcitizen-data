from numpy import nan, full, power, ones
from scdata.device.process import is_within_circle

def poly_ts(dataframe, **kwargs):
    """
    Calculates the a polinomy based on channels.
    Parameters
    ----------
        channels: list of strings
            list containing channels
        coefficients: list or np array
            list containing coefficients
        exponents: list or np array
            list containing exponents
        extra_term: float
            0
            Independent term
    Returns
    -------
        result = sum(coefficients[i]*channels[i]^exponents[i] + extra_term)
    """
    
    if 'channels' not in kwargs: return None
    else: channels = kwargs['channels']
    n_channels = len(channels)

    result = 0 * dataframe[channels[0]].copy()

    if 'coefficients' not in kwargs: coefficients = ones(n_channels)
    else: coefficients = kwargs['coefficients']
    
    if 'exponents' not in kwargs: exponents = ones(n_channels)
    else: exponents = kwargs['exponents']
    
    if 'extra_term' not in kwargs: extra_term = 0
    else: extra_term = kwargs['extra_term']

    # Do it in pandas to ensure index and na handling
    for i in range(n_channels):
        result += power(dataframe[channels[i]], exponents[i]) * coefficients[i]

    return result + extra_term

def clean_ts(dataframe, **kwargs):
    """
    Cleans the time series measurements sensors, by filling the out of band values with NaN
    Parameters
    ----------
        name: string
            column to clean to apply.
        limits: list, optional 
            (0, 99999)
            Sensor limits. The function will fill with NaN in the values that exceed the band
        window_size: int, optional 
            3
            If not None, will smooth the time series by applying a rolling window of that size
        window_type: str, optional
            None
            Accepts arguments in the list of windows for scipy.signal windows:
            https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions
            Default to None implies normal rolling average
    Returns
    -------
        pandas series containing the clean
    """

    if 'name' not in kwargs: return None
    if kwargs['name'] not in dataframe: return None

    result = dataframe[kwargs['name']].copy()

    # Limits
    if 'limits' in kwargs: lower_limit, upper_limit = kwargs['limits'][0], kwargs['limits'][1]
    else: lower_limit, upper_limit = 0, 99999

    result[result > upper_limit] = nan
    result[result < lower_limit] = nan
     
    # Smoothing
    if 'window_size' in kwargs: window = kwargs['window_size'] 
    else: window = 3

    if 'window_type' in kwargs: win_type = kwargs['window_type']
    else: win_type = None

    if window is not None:
        result.rolling(window = window, win_type = win_type).mean()

    return result

def merge_ts(dataframe, **kwargs):
    """
    Merges readings from sensors into one clean ts. The function checks the dispersion and 
    picks the desired one (min, max, min_nonzero, avg)
    Parameters
    ----------
        names: list of strings
            List of sensors to merge into one. Currently only support two ts.
        pick: string
            'min'
            One of the following 'min', 'max', 'avg', 'min_nonzero'
            Which one two pick in case of high deviation between the metrics. Picks the avg 
            otherwise
        factor: float (factor > 0)
            0.3
            Maximum allowed deviation of the difference with respect to the each of signals.
            It creates a window of [factor*signal_X, -factor*signal_X] for X being each signal
            out of which there will be a flag where one of the signals will be picked. This 
            factor should be set to a value that is similar to the sensor typical deviation
        Same parameters as clean_ts apply below:
        limits: list, optional 
            (0, 99999)
            Sensor limits. The function will fill with NaN in the values that exceed the band
        window_size: int, optional 
            3
            If not None, will smooth the time series by applying a rolling window of that size
        window_type: str, optional
            None
            Accepts arguments in the list of windows for scipy.signal windows:
            https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions
            Default to None implies normal rolling average
    Returns
    -------
        pandas series containing the clean
    """

    df = dataframe.copy()

    # Set defaults
    if 'names' not in kwargs: return None
    if 'pick' not in kwargs: pick = 'min'
    else: pick = kwargs['pick']
    if 'factor' not in kwargs: factor = 0.3
    else: factor = kwargs['factor']

    # Clean them
    for name in kwargs['names']: 
        subkwargs = {'name': name, 
                    'limits': kwargs['limits'], 
                    'window_size': kwargs['window_size'], 
                    'window_type': kwargs['window_type']
                    }
        df[name + '_CLEAN'] = clean_ts(df, **subkwargs)

    
    df['flag'] = full((df.shape[0], 1), False, dtype=bool)
    df['diff'] = df[kwargs['names'][0] + '_CLEAN'] - df[kwargs['names'][1] + '_CLEAN']

    lnames = []
    # Flag them
    for name in kwargs['names']:
        df['flag'] |= (df['diff'] > factor*df[name + '_CLEAN'])
        df['flag'] |= (df['diff'] < -factor*df[name + '_CLEAN'])
        lnames.append(name + '_CLEAN')

    df['result'] = df.loc[:, lnames].mean(skipna=True, axis = 1)
    
    # Pick
    if pick == 'min': 
        df.loc[df['flag'] == True, 'result'] = df.loc[df['flag'] == True, lnames].min(skipna=True, axis = 1)
    elif pick == 'max':
        df.loc[df['flag'] == True, 'result'] = df.loc[df['flag'] == True, lnames].max(skipna=True, axis = 1)
    # elif pick == 'min_nonzero':
    #     df['result'] = df.loc[df['flag'] == True, kwargs['names']].min(skipna=True, axis = 1)

    return df['result']

def rolling_avg(dataframe, **kwargs):
    """
    Performs pandas.rolling with input
    Parameters
    ----------
        name: string
            column to clean to apply.
        window_size: int, optional 
            3
            If not None, will smooth the time series by applying a rolling window of that size
        window_type: str, optional
            None
            Accepts arguments in the list of windows for scipy.signal windows:
            https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions
            Default to None implies normal rolling average
    Returns
    -------
        pandas series containing the rolling average
    """   

    if 'name' not in kwargs: return None

    result = dataframe[kwargs['name']].copy()

    # Smoothing
    if 'window_size' in kwargs: window = kwargs['window_size'] 
    else: window = 3

    if 'window_type' in kwargs: win_type = kwargs['window_type']
    else: win_type = None

    if 'type' in kwargs: 
        if kwargs['type'] == 'mean': return result.rolling(window = window, win_type = win_type).mean()
        if kwargs['type'] == 'max': return result.rolling(window = window, win_type = win_type).max()
        if kwargs['type'] == 'min': return result.rolling(window = window, win_type = win_type).min()
    else:
        return result.rolling(window = window, win_type = win_type).mean()

def geo_located(dataframe, **kwargs):
    """
    Performs pandas.rolling with input
    Parameters
    ----------
        within: tuple
            Empty tuple
            Gets the devices within a circle center on lat, long with a radius_meters
            within = tuple(lat, long, radius_meters)
        lat_name: str, optional
            GPS_LAT
            Column name for latitude in dataframe
        long_name: str, optional
            GPS_LONG
            Column name for long in dataframe
    Returns
    -------
        pandas series containing bool defining wether or not each dataframe[:, [lat_name, long_name]] are within circle of basepoint(lat, long)
    """

    result = dataframe.copy()

    if 'within' not in kwargs: return None

    if 'lat_name' not in kwargs: lat_name = 'GPS_LAT'
    if 'long_name' not in kwargs: long_name = 'GPS_LONG'

    result['within'] = result.apply(lambda x: is_within_circle(x, kwargs['within'], lat_name, long_name), axis=1)

    return result['within']