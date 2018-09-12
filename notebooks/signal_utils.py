import numpy as np
from math import sqrt

def metrics(reference, estimation):
    metrics_dict = dict()
    
    # Average
    avg_ref = np.mean(reference)
    avg_est = np.mean(estimation)
    metrics_dict['avg_ref'] = avg_ref
    metrics_dict['avg_est'] = avg_est

    # Standard deviation
    sigma_ref = np.std(reference)
    sigma_est = np.std(estimation)
    metrics_dict['sig_ref'] = sigma_ref
    metrics_dict['sig_est'] = sigma_est
    
    # Bias
    bias = avg_est-avg_ref
    normalised_bias = float((avg_est-avg_ref)/sigma_ref)
    metrics_dict['bias'] = bias
    metrics_dict['normalised_bias'] = normalised_bias
    
    # Normalized std deviation
    sigma_norm = sigma_est/sigma_ref
    sign_sigma = (sigma_est-sigma_ref)/(abs(sigma_est-sigma_ref))
    metrics_dict['sigma_norm'] = sigma_norm
    metrics_dict['sign_sigma'] = sign_sigma

    # R2
    SS_Residual = sum((estimation-reference)**2)
    SS_Total = sum((reference-np.mean(reference))**2)
    rsquared = max(0, 1 - (float(SS_Residual))/SS_Total)
    metrics_dict['rsquared'] = rsquared

    # RMSD
    RMSD = sqrt((1./len(reference))*SS_Residual)
    RMSD_norm_unb = sqrt(1+np.power(sigma_norm,2)-2*sigma_norm*sqrt(rsquared))
    metrics_dict['RMSD'] = RMSD
    metrics_dict['RMSD_norm_unb'] = RMSD_norm_unb
    
    return metrics_dict

def detect_peak(signal):
    result = np.zeros(signal.shape)
    for i in range(len(signal)-1):
        if signal[i+1] > signal[i]: result[i] = 1
        elif signal[i+1] < signal[i]: result[i] = -1
        elif signal[i+1] == signal[i]: result[i] = 0
    return result

def count_peak(signal, acum = False):
    count_pos = 0
    count_neg = 0
    last_peak_sign = 1
    peak = detect_peak(signal)
    result = np.zeros(signal.shape)
    for i in range(len(signal)-1):
        last_peak_sign = 1
        if acum:
            if peak[i] == 1:
                last_peak_sign = 1
                result[i] = 0
            elif peak[i] == -1:
                last_peak_sign = -1
                result[i] = 0
            elif peak[i] == 0: 
                if i == 0: result[i] = 1
                else: result[i] = result[i-1] + last_peak_sign
        else:
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
    dataframe_combined = combine_data(_readings[_reading]['devices'], False)
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

import plotly.tools as tls
import plotly as ply
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

def plot_oneshots(readings, channels, device_one_shot):
    for reading in readings:
        devices = readings[reading]['devices'].keys()
        
        # Combine the devices for each reading
        dataframe_combined = combine_data(readings[reading]['devices'], False)
        readings[reading]['devices']['combined'] = dict()
        readings[reading]['devices']['combined']['data'] = dict()
        readings[reading]['devices']['combined']['data'] = dataframe_combined
        
        # Check if the first channel is measuring
        dataframe_combined['measuring'] = 1.0-1.0*np.isnan(readings[reading]['devices']['combined']['data'][channels[0] + '_' + device_one_shot])
        
        # Count the peaks on measuring channel (basicly if it's on) and accumulate how much time has it been measuring
        dataframe_combined['change'] = count_peak(dataframe_combined['measuring'], True)
        
        # Group it be 'change'
        df = [x for _, x in dataframe_combined.groupby('change')]
                
        fig1 = tls.make_subplots(rows=len(channels), cols=1, shared_xaxes=True)
        
        for channel in channels:
            
            indexAB = list() 
            meanAB = list()
            upperAB = list()
            lowerAB = list()
        
            for i in range(len(df)):
                # PM 1.0
                dataframePM = df[i].loc[df[i]['measuring']==1]
                dataA = dataframePM.loc[:,channel + '_' + devices[0]]
                dataB = dataframePM.loc[:,channel + '_' + devices[1]]
                dataAB_y = (dataB - dataA)/dataB
                indexAB.append(i)
                meanAB.append(np.mean(dataAB_y))
                upperAB.append(np.mean(dataAB_y)+np.std(dataAB_y))
                lowerAB.append(np.mean(dataAB_y)-np.std(dataAB_y))
                dataAB_x = i*np.ones(dataAB_y.shape)
                
                fig1.append_trace({'x': dataAB_x, 
                        'y': dataAB_y, 
                        'type': 'scatter', 
                        'name': 'ERROR',
                        'mode': 'markers',
                        'marker': dict(
                            size = 5,
                            color = 'rgba(255, 10, 0, .7)',
                            )}, channels.index(channel)+1 , 1)
                              

                
            fig1.append_trace({'x': indexAB, 
                    'y': meanAB, 
                    'type': 'scatter', 
                    'name': 'Mean', 
                    'mode': 'lines',
                    'marker': dict(
                            color = 'rgba(0, 10, 255, 1)',
                            )}, channels.index(channel)+1 , 1)
            
            fig1.append_trace({'x': indexAB, 
                    'y': upperAB, 
                    'type': 'scatter', 
                    'name': 'Upper Bound',
                    'mode': 'lines',
                    'marker': dict(
                            color = 'rgba(10, 255, 0, .7)',
                            )}, channels.index(channel)+1 , 1)
            
            fig1.append_trace({'x': indexAB, 
                    'y': lowerAB, 
                    'type': 'scatter', 
                    'name': 'Lower Bound',
                    'mode': 'lines',
                    'marker': dict(
                            color = 'rgba(10, 255, 0, .7)',
                            )}, channels.index(channel)+1 , 1)
            
            fig1['layout'].update(height = 1400,
                    showlegend=False,
                    xaxis=dict(title='Measurement Time (s)'),
                    title = reading)
            fig1['layout']['yaxis'+str(channels.index(channel)+1)].update(title=channel + ' Relative Error')

        ply.offline.iplot(fig1)

def minRtarget(targetR):
    return sqrt(1+ np.power(targetR,2)-2*np.power(targetR,2))
