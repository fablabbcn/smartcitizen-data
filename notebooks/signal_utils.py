import numpy as np


def detect_peak(signal):
    result = np.zeros(signal.shape)
    for i in range(len(signal)-1):
        if signal[i+1] > signal[i]: result[i] = 1
        elif signal[i+1] < signal[i]: result[i] = -1
        elif signal[i+1] == signal[i]: result[i] = 0
    return result

def count_peak(signal):
    count_pos = 0
    count_neg = 0
    peak = detect_peak(signal)
    result = np.zeros(signal.shape)
    for i in range(len(signal)-1):
        if peak[i] == 1: 
            count_pos = count_pos + 1
            result[i] = count_pos
        elif peak[i] == -1: 
            count_neg = count_neg + 1
            result[i] = -count_neg
        elif peak[i] == 0: 
            if i == 0: result[i] = 1
            else: result[i] = result[i-1]
    return result       

from signal_utils import *
from test_utils import combine_data
from dateutil import relativedelta
import pandas as pd

def split_agnostisise(_readings, _reading, _channel):        
    begining_date = '2001-01-01 00:00:00+02:00'
    print _readings[_reading]['devices'].keys()
    dataframe_combined = combine_data(_readings[_reading]['devices'])
    #readings[_reading]['devices']['merged'] = dict()
    #readings[_reading]['devices']['merged']['data'] = dict()
    #readings[_reading]['devices']['merged']['data'] = dataframe_combined
    
    #dataframe = readings['2018-07_INT_TEMP_CALIB_CASE_BOTH_25degC']['devices']['merged']['data']
    dataframe_combined['change'] = count_peak(dataframe_combined[_channel])

    df = [x for _, x in dataframe_combined.groupby('change')]
    init_date = pd.to_datetime(begining_date).tz_localize('UTC').tz_convert('UTC')
    dataframeAgnostic = pd.DataFrame()
    for i in range(len(df)):
        min_date= pd.to_datetime(df[i].index.min()).tz_convert('UTC')
    
        # print init_date
        # print min_date
        # print relativedelta.relativedelta(min_date, init_date)
        years = relativedelta.relativedelta(min_date, init_date).years
        months = relativedelta.relativedelta(min_date, init_date).months
        days = relativedelta.relativedelta(min_date, init_date).days
        hours = relativedelta.relativedelta(min_date, init_date).hours
        minutes = relativedelta.relativedelta(min_date, init_date).minutes
        seconds = relativedelta.relativedelta(min_date, init_date).seconds
        
        df[i].index =  df[i].index - pd.DateOffset(years = years, months = months, days = days, hours = hours, minutes = minutes, seconds = seconds)
        
        if df[i].loc[:, _channel].mean() > 0.5: prepend = '_ON_' + str(df[i].loc[:,'change'].mean())
        else: prepend = '_OFF_' + str(df[i].loc[:,'change'].mean())
        new_names = list()
        
        for name in df[i].columns:
            # print name
            new_names.append(name + prepend)
        
        df[i].columns = new_names
        dataframeAgnostic = dataframeAgnostic.combine_first(df[i])
    
    return dataframeAgnostic 
