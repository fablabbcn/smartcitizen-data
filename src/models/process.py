import numpy as np
import math
import pandas as pd
import sys

from src.saf import std_out
from dateutil import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plot
from xgboost import XGBRegressor

class LazyCallable(object):
    '''
        Adapted from Alex Martelli's answer on this post on stackoverflow:
        https://stackoverflow.com/questions/3349157/python-passing-a-function-name-as-an-argument-in-a-function
    '''
    def __init__(self, name):
        self.n = name
        self.f = None
    def __call__(self, *a, **k):
        if self.f is None:
            std_out(f"Loading {self.n.rsplit('.', 1)[1]} from {self.n.rsplit('.', 1)[0]}")
            modn, funcn = self.n.rsplit('.', 1)
            if modn not in sys.modules:
                __import__(modn)
            self.f = getattr(sys.modules[modn], funcn)
        return self.f(*a, **k)

### ---------------------------------------
### --------------PROCESSES----------------
### ---------------------------------------
'''
All functions below are meant to return an np.array object after receiving a pd.DataFrame.
You can implement the process and then use a lazy_callable instance
to invoke the function by passing the corresponding *args to it.
'''

def hello_world(string):
    '''
    Example of lazy callable function
    '''
    print (string)
    return 82

def sum(dataframe, *args):
    '''
    Example of lazy callable function returning the sum of two channels in a pandas dataframe
    '''
    df = dataframe.copy()
    series = df[args[0]] + df[args[1]]
    return series

# TODO
def baseline_dcalc(dataframe, **kwargs):
    return dataframe[kwargs['working']]
# TODO
def co_calc(dataframe, **kwargs):
    return dataframe[kwargs['working']]

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

    result = dataframe[kwargs['name']].copy()

    # Limits
    if 'limits' in kwargs: lower_limit, upper_limit = kwargs['limits'][0], kwargs['limits'][1]
    else: lower_limit, upper_limit = 0, 99999

    result[result >= upper_limit] = np.nan
    result[result <= lower_limit] = np.nan
     
    # Smoothing
    if 'window_size' in kwargs: window = kwargs['window_size'] 
    else: window = 3

    if 'window_type' in kwargs: win_type = kwargs['window_type']
    else: win_type = None

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

    
    df['flag'] = np.full((df.shape[0], 1), False, dtype=bool)
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

def rolling_avg(dataframe, *kwargs):
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

    result.rolling(window = window, win_type = win_type).mean()

    return result
    
# TODO
def absolute_humidity(dataframe, **kwargs):
    """
    Calculate Absolute humidity based on vapour equilibrium
    Parameters
    ----------
        temperature: string
            'TEMP'
            Name of the column in the daframe for temperature in degC
        rel_h: string
            'HUM'
            Name of the column in the daframe for relative humidity in %rh
        pressure: string
            'PRESS'
            Name of the column in the daframe for atmospheric pressure in mbar
    Returns
    -------
        pandas series containing the absolute humidity calculation in mg/m3?
    """
    # Check
    if 'temperature' not in kwargs: return None
    if 'rel_h' not in kwargs: return None
    if 'pressure' not in kwargs: return None

    if kwargs['temperature'] not in dataframe.columns: return None
    if kwargs['rel_h'] not in dataframe.columns: return None
    if kwargs['pressure'] not in dataframe.columns: return None

    temp = dataframe[kwargs['temperature']].values
    rel_h = dataframe[kwargs['rel_h']].values
    press = dataframe[kwargs['pressure']].values

    # _Temp is temperature in degC, _Press is absolute pressure in mbar
    vap_eq = (1.0007 + 3.46*1e-6*press)*6.1121*np.exp(17.502*temp/(240.97+temp))

    abs_humidity = rel_h * vap_eq

    return abs_humidity

### ---------------------------------------
### ----------------OLDIES-----------------
### ---------------------------------------

def exponential_smoothing(series, alpha = 0.5):
    '''
        Input:
            series - pandas series with timestamps
            alpha - float [0.0, 1.0], smoothing parameter
        Output: 
            smoothed series
    '''
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def box_convolution(series, box_pts):
    '''
        Implements 1-d square-box filtering on an input signal, using padding on both sides of the signal
        series: pandas series
        box_pts: must be an odd number
    '''
    
    half_window = (box_pts -1) // 2
    try:
        box = np.ones(box_pts)/box_pts
        firstvals = series[0] - np.abs( series[1:half_window+1][::-1] - series[0] )
        lastvals = series[-1] + np.abs(series[-half_window-1:-1][::-1] - series[-1])
        series = np.concatenate((firstvals, list(map(float, series)), lastvals))
        series_smooth = np.convolve(series, box, mode='valid')
        return series_smooth
    except:
        pass
    return np.ones(len(series))

def derivative(y, x):
    dx = np.zeros(x.shape, np.float)
    dy = np.zeros(y.shape, np.float)
    
    if isinstance(x, pd.DatetimeIndex):
        print ('x is of type DatetimeIndex')
        
        for i in range(len(x)-1):
            dx[i] =  (x[i+1]-x[i]).seconds
            
        dx [-1] = np.inf
    else:
        dx = np.diff(x)

    dy[0:-1] = np.diff(y)/dx[0:-1]
    dy[-1] = (y[-1] - y[-2])/dx[-1]
    result = dy
    return result

def time_derivative(series, window = 1):

    return series.diff(periods = window)/(2*series.index.to_series().diff(periods = window).dt.total_seconds())

def time_diff(series, window = 1):
    return series.index.to_series().diff(periods = window).dt.total_seconds()

def gradient(series, raster):
    return np.gradient(series, raster*2)


def maxer(series, val):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<=val and not series[i] == np.nan): result[i] = val
        elif (series[i]>val and not series[i] == np.nan): result[i] = series[i]
        elif (math.isnan(series[i])): result[i] = np.nan
    return result

def maxer_hist(series, val, hist):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<=val and not series[i] == np.nan): result[i] = hist
        elif (series[i]>val and not series[i] == np.nan): result[i] = series[i]
        elif (math.isnan(series[i])): result[i] = np.nan
    return result

def miner(series, val):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<=val and not series[i] == np.nan): result[i] = series[i]
        elif (series[i]>val and not series[i] == np.nan): result[i] = val
        elif (math.isnan(series[i])): result[i] = np.nan
    return result

def greater(series, val):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<=val and not series[i] == np.nan): result[i] = False
        elif (series[i]>val and not series[i] == np.nan): result [i] = True
        elif (math.isnan(series[i])): result[i] = result[i-1]   
    return result

def greaterequal(series, val):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<val and not series[i] == np.nan): result[i] = False
        elif (series[i]>=val and not series[i] == np.nan): result [i] = True
        elif (math.isnan(series[i])): result[i] = result[i-1]
    return result

def lower (series, val):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<val and not series[i] == np.nan): result[i] = True
        elif (series[i]>=val and not series[i] == np.nan): result [i] = False
        elif (math.isnan(series[i])): result[i] = result[i-1]   
    return result

def lowerequal(series, val):
    result = np.zeros(len(series))
    for i in range(len(series)):
        if (series[i]<=val and not series[i] == np.nan): result[i] = True
        elif (series[i]>val and not series[i] == np.nan): result [i] = False
        elif (math.isnan(series[i])): result[i] = result[i-1] 
    return result
            
def exponential_func(x, a, b, c):
     return a * np.exp(b * x) + c

def evaluate(predictions, original):
    errors = abs(predictions - original)
    max_error = max(errors)
    rerror = np.maximum(np.minimum(np.divide(errors, original),1),-1)
    
    mape = 100 * np.mean(rerror)
    accuracy = 100 - mape
    
    return max_error, accuracy