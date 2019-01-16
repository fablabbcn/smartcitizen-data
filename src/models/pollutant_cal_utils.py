import matplotlib
import matplotlib.pyplot as plot                  # plots
import seaborn as sns                            # more plots
sns.set(color_codes=True)
matplotlib.style.use('seaborn-whitegrid')

import plotly as ply                             # even more plots
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Layout
import plotly.tools as tls

import pandas as pd
import numpy as np

import datetime
from scipy.stats.stats import linregress   
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
from dateutil import relativedelta
from scipy.optimize import curve_fit

from src.data.data_utils import getCalData
from src.data.test_utils import *
from src.models.formula_utils import exponential_smoothing, miner, maxer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt, isnan

# AlphaDelta PCB factor
factorPCB = 6.36

# Background Concentration (model assumption) - (from Modelling atmospheric composition in urban street canyons - Vivien Bright, William Bloss and Xiaoming Cai)
backgroundConc_CO = 0 # ppm
backgroundConc_NO2 = 8 # ppb
backgroundConc_OX = 40 # ppb

# Overlap in hours for each day (index = [day(i)-overlapHours, day(i+1)+overlapHours])
overlapHours = 2 

# Filter Smoothing 
filterExpSmoothing = 0.2

# Range of deltas
deltasMICS = np.arange(1,200,1)

# Units Look Up Table - ['Pollutant', unit factor from ppm to target 1, unit factor from ppm to target 2]
alphaUnitsFactorsLUT = (['CO', 1, 0],
                        ['NO2', 1000, 0],
                        ['O3', 1000, 1000])

micsUnitsFactorsLUT = (['CO', 1],
                        ['NO2', 1000])

# Alphasense ID table (Slot, Working, Auxiliary)
as_ids_table = ([1,'64','65'], 
             [2,'61','62'], 
             [3,'67','68'])

# External temperature table (this table is by priority)
th_ids_table = (['EXT_DALLAS','96',''], 
                 ['EXT_SHT31','79', '80'], 
                 ['SENSOR_TEMPERATURE','55','56'],
                 ['GASESBOARD_TEMPERATURE','79', '80'])

def ExtractBaseline(_data, _delta):
    '''
        Input:
            _data: dataframe containing signal to be baselined and index
            _delta : float for delta time (N periods)
        Output:
            result: vector containing baselined values
    ''' 
    
    result = np.zeros(len(_data))
    name = []
    for n in range(0, len(_data)):
        minIndex = max(n-_delta,0)
        maxIndex = min(n+_delta, len(_data))
        
        chunk = _data.values[minIndex:maxIndex]
        result[n] = min(chunk)

    return result

def findMax(_listF):
    '''
        Input: list to obtain maximum value
        Output: value and index of maximum in the list
    '''
    
    valMax=np.max(_listF)
    indexMax = np.argmax(_listF)
    # print 'Max Value found at index: {}, with value: {}'.format(indexMax, valMax)
    
    return valMax, indexMax

def exponential_func(x, a, b, c):
     return a * np.exp(b * x) + c

def createBaselines(_dataBaseline, _numberDeltas, _type_regress = 'linear', _plots = False, _verbose = False):
    '''
        Input:
            _dataBaseline: dataframe containing signal and index to be baselined
            _type_regress= 'linear', 'exponential', 'best' (based on p_value of both)
            _numberDeltas : vector of floats for deltas (N periods)
            _plots:  display plots or not
            _verbose: print info or not
            _type_regress: regression type (linear, exp, ... )
        Output:
            baseline: pandas dataframe baseline
        TODO:
            implement other types of regression
    '''
    
    resultData = _dataBaseline.copy()

    name = resultData.iloc[:,0].name
    pearsons =[]
    
    for delta in _numberDeltas:
        resultData[(name +'_' +str(delta))] = ExtractBaseline(resultData.iloc[:,0], delta)
        slope, intercept, r_value, p_value, std_err = linregress(np.transpose(resultData[(name +'_' +str(delta))]), np.transpose(resultData.iloc[:,1].values))
        pearsons.append(r_value)
    
    ## Find Max in the pearsons - correlation can be negative, so use absolute of the pearson
    valMax, indexMax = findMax(np.abs(pearsons))
        
    ## Find regression between _dataCorr
    baseline = pd.DataFrame(index = _dataBaseline.index)
    if _type_regress == 'linear':
        ## Fit with y = A + Bx
        slope, intercept, r_value, p_value, std_err = linregress(np.transpose(resultData.iloc[:,1].values),resultData[(name + '_'+str(_numberDeltas[indexMax]))])
        baseline[(name + '_' + 'baseline_' +  _type_regress)] = intercept + slope*resultData.iloc[:,1].values
        # print (r_value)
    elif _type_regress == 'exponential':
        ## Fit with y = Ae^(Bx) -> logy = logA + Bx
        logy = np.log(resultData[(name + '_'+str(_numberDeltas[indexMax]))])
        slope, intercept, r_value, p_value, std_err = linregress(np.transpose(resultData.iloc[:,1].values), logy)
        baseline[(name + '_' + 'baseline_' +  _type_regress)] = exponential_func(np.transpose(resultData.iloc[:,1].values), np.exp(intercept), slope, 0)
        # print (r_value)
    elif _type_regress == 'best':
        ## Find linear r_value
        slope_lin, intercept_lin, r_value_lin, p_value_lin, std_err_lin = linregress(np.transpose(resultData.iloc[:,1].values),resultData[(name + '_'+str(_numberDeltas[indexMax]))])
        
        ## Find Exponential r_value
        logy = np.log(resultData[(name + '_'+str(_numberDeltas[indexMax]))])
        slope_exp, intercept_exp, r_value_exp, p_value_exp, std_err_exp = linregress(np.transpose(resultData.iloc[:,1].values), logy)
        
        ## Pick which one is best
        if ((not isnan(r_value_exp)) and (not isnan(r_value_lin))):
            if r_value_lin > r_value_exp:
                if _verbose:
                    print ('Using linear regression')
                baseline[(name + '_' + 'baseline_' +  _type_regress)] = intercept_lin + slope_lin*resultData.iloc[:,1].values
            else:
                if _verbose:
                    print ('Using exponential regression')
                baseline[(name + '_' + 'baseline_' +  _type_regress)] = exponential_func(np.transpose(resultData.iloc[:,1].values), np.exp(intercept_exp), slope_exp, 0)
        elif not isnan(r_value_lin):
            if _verbose:
                print ('Using linear regression')
            baseline[(name + '_' + 'baseline_' +  _type_regress)] = intercept_lin + slope_lin*resultData.iloc[:,1].values
        elif not isnan(r_value_exp):
            if _verbose:
                print ('Using exponential regression')
            baseline[(name + '_' + 'baseline_' +  _type_regress)] = exponential_func(np.transpose(resultData.iloc[:,1].values), np.exp(intercept_exp), slope_exp, 0)
        
    if _plots == True:
        with plot.style.context('seaborn-white'):
            fig1, (ax1, ax2) = plot.subplots(nrows=1, ncols=2, figsize=(20,8))
            
            ax1.plot(resultData.iloc[:,1].values, resultData[(name + '_'+str(_numberDeltas[indexMax]))], label = 'Baseline', linestyle='-', linewidth=0, marker='o')
            ax1.plot(resultData.iloc[:,1].values, baseline[(name + '_' + 'baseline_' +  _type_regress)] , label = 'Regressed value', linestyle='-', linewidth=1, marker=None)
            legend = ax1.legend(loc='best')
            ax1.set_xlabel(resultData.iloc[:,1].name)
            ax1.set_ylabel('Regression values')
            ax1.grid(True)
            
            ax2.plot(resultData.iloc[:,0].index, resultData.iloc[:,0].values, label = "Actual", linestyle=':', linewidth=1, marker=None)
            #[ax2.plot(resultData.index, resultData[(name +'_' +str(delta))].values, label="Delta {}".format(delta), marker=None,  linestyle='-', linewidth=1) for delta in _numberDeltas]
            ax2.plot(baseline.index, baseline.values, label='Baseline', marker = None)

            ax2.axis('tight')
            ax2.legend(loc='best')
            ax2.set_title("Baseline Extraction")
            ax2.grid(True)
            
            ax22 = ax2.twinx()
            ax22.plot(resultData.index, resultData.iloc[:,1].values, color = 'red', label = resultData.iloc[:,1].name, linestyle='-', linewidth=1, marker=None)
            ax22.set_ylabel(resultData.iloc[:,1].name, color = 'red')
            ax22.tick_params(axis='y', labelcolor='red')
            
            fig2, ax3 = plot.subplots(figsize=(20,8)) # two axes on figure
            ax3.plot(_numberDeltas, pearsons)
            ax3.axis('tight')
            ax3.set_title("R2 vs. Delta")
            ax3.set_xlabel('Delta')
            ax3.set_ylabel('R2')
            ax3.grid(True)

    return baseline, indexMax

def decompose(_data, plots = False):
    '''
            Function to decompose a signal into it's trend and normal variation
            Input:
                _data: signal to decompose
                plots: print plots or not (default False)
            Output:
                DataDecomp = _data - slope*_data.index
                slope, intercept = linear regression coefficients
    '''
    indexDecomp = np.arange(len(_data))

    slope, intercept, r_value, p_value, std_err = linregress(indexDecomp, np.transpose(_data.values))
    dataDecomp=pd.DataFrame(index = _data.index)
    name = _data.name
    result = []
    
    for n in range(len(_data)):
        result.append(float(_data.values[n]-slope*n))
    dataDecomp[(name + '_' + '_flat')] = result
    
    trend = slope*indexDecomp + intercept
    if plots == True:
        
        with plot.style.context('seaborn-white'):
            fig, ax = plot.subplots(figsize=(20,10))
            ax.plot(_data.index, _data.values, label = "Actual", marker = None)
            ax.plot(_data.index, dataDecomp[(name + '_' +'_flat')], marker = None, label = 'Flattened')
            ax.plot(_data.index, trend, label = 'Trend')
            ax.legend(loc="best")
            ax.axis('tight')
            ax.set_title("Signal Decomposition - "+ name)
            ax.set_xlabel('Index')
            ax.set_ylabel('Signal')
            ax.grid(True)
            
    return dataDecomp, slope, intercept

def calculateBaselineDay(_dataFrame, _typeSensor, _listNames, _deltas, _type_regress, _trydecomp = False, _plots = False, _verbose = True):
    '''
        Function to calculate day-based baseline corrections
        Input:
            _dataFrame: pandas dataframe with datetime index containing 1 day of measurements + overlap
            _listNames: list containing column names of /WE / AE / TEMP / HUM) or (MICS / TEMP / HUM)
            _baselined: channel to calculate the baseline of
            _baseliner: channel to use as input for the baselined (baselined = f(baseliner)) 
            _type_regress: type of regression to perform (linear, exponential, best)
            // NOT USED - _trydecomp: try trend decomposition (not needed . remove it)
            _plot: plot analytics or not
            _verbose: print analytics or not
        Output:
            _data_baseline: dataframe with baseline
            _baseline_corr: metadata containing analytics for long term analysis
    '''
    
    ## Create Baselines

    dataframeCalc = _dataFrame.copy()

    data_baseline, indexMax = createBaselines(dataframeCalc, _deltas, _type_regress, _plots, _verbose)

    if _typeSensor == 'alphasense': 

        ## Un-pack list names
        alphaW, alphaA, temp, hum = _listNames

        ## Correlation between Baseline and original auxiliary
        slopeAuxBase, interceptAuxBase, rAuxBase, pAuxBase, std_errAuxBase = linregress(np.transpose(data_baseline.values), np.transpose(dataframeCalc.iloc[:,1].values))

        # print 'Working', dataframeCalc.iloc[:,0].name
        # print 'Baseliner', dataframeCalc.iloc[:,1].name
        # Add metadata for further research
        deltaAuxBase_avg = np.mean(data_baseline.values-dataframeCalc.iloc[:,1].values)
        ratioAuxBase_avg = np.mean(data_baseline.values/dataframeCalc.iloc[:,1].values)
       
        # Pre filter based on the metadata itself
        if slopeAuxBase > 0 and rAuxBase > 0.3:
            valid = True
        else:
            valid = False
       
        CorrParams = {}

        CorrParams['slopeAuxBase'] = slopeAuxBase
        CorrParams['rAuxBase'] = rAuxBase
        CorrParams['deltaAuxBase_avg'] = deltaAuxBase_avg
        CorrParams['ratioAuxBase_avg'] = ratioAuxBase_avg
        CorrParams['indexMax'] = indexMax
        CorrParams['0_valid'] = valid
    
        if _verbose == True:
            
            print ('-------------------')
            print ('Auxiliary Electrode')
            print ('-------------------')
            print ('Correlation coefficient of Baseline and Original auxiliary: {}'.format(rAuxBase))
            
            print ('Average Delta: {} \t Average Ratio: {}'.format(deltaAuxBase_avg, deltaAuxBase_avg))

        if _plots == True:
            with plot.style.context('seaborn-white'):
                
                fig2, ax3 = plot.subplots(figsize=(20,8))
 
                ax3.plot(data_baseline.index, data_baseline.values, label='Baseline', marker = None)
                ax3.plot(dataframeCalc.index, dataframeCalc.iloc[:,0], label='Original Working', marker = None)
                ax3.plot(dataframeCalc.index, dataframeCalc.iloc[:,1], label='Original Auxiliary', marker = None)

                ax3.legend(loc="best")
                ax3.axis('tight')
                ax3.set_title("Baseline Not Compensated")
                ax3.set(xlabel='Time', ylabel='Ouput-mV')
                ax3.grid(True)
                 
                # fig3, ax7 = plot.subplots(figsize=(20,8))
                
                # ax7.plot(_dataFrame[temp], _dataFrame[alphaW], label='W - Raw', marker='o',  linestyle=None, linewidth = 0)
                # ax7.plot(_dataFrame[temp], _dataFrame[alphaA], label ='A - Raw', marker='v', linewidth=0)
                
                # ax7.legend(loc="best")
                # ax7.axis('tight')
                # ax7.set_title("Output vs. Temperature")
                # ax7.set(xlabel='Temperature', ylabel='Ouput-mV')
                # ax7.grid(True)

    elif _typeSensor == 'mics':
        ## Un-pack list names
        mics_resist, temp, hum = _listNames

        baselineCorr = list()
        baselineCorr.append(indexMax)

    return data_baseline, CorrParams

def findDates(_dataframe):
    '''
        Find minimum, maximum dates in the dataframe and the amount of days in between
        Input: 
            _dataframe: pandas dataframe with datetime index
        Output: 
            rounded up min day, floor max day and number of days between the min and max dates
    '''
    range_days = (_dataframe.index.max()-_dataframe.index.min()).days
    # min_date_df = _dataframe.index.min().ceil('D')
    # max_date_df = _dataframe.index.max().floor('D')
    min_date_df = _dataframe.index.min().floor('D')
    max_date_df = _dataframe.index.max().ceil('D')
    
    return min_date_df, max_date_df, range_days

def calculatePollutantsAlpha(_dataframe, _pollutantTuples, _append, _refAvail, _dataframeRef, _overlapHours = 0, _type_regress = 'best', _filterExpSmoothing = 0.2, _trydecomp = False, _plotsInter = False, _plotResult = True, _verbose = False, _printStats = False, _calibrationDataPath = '', _currentSensorNames = ''):
    '''
        Function to calculate alphasense pollutants with baseline technique
        Input:
            _dataframe: pandas dataframe from
            _pollutantTuples: list of tuples containing: 
                '[(_pollutant, 
                _sensorID, 
                calibration_method: 
                    'classic'
                    'baseline', 
                baseline_type: 'baseline using auxiliary, temperature or humidity'
                    ''
                    'single_aux'
                    'single_temp'
                    'single_hum'
                sensor_slot), 
                ...]'
            _append: suffix to the new channel name (pollutant + append)
            _refAvail: True or False if there is a reference available
            _dataframeRef: reference dataframe if available
            _overlapHours: number of hours to overlap over the day -> -_overlapHours+day:day+1+_overlapHours
            _type_regress = type of regression for baseline (best, exponential or linear)
            _filterExpSmoothing = alpha parameter for exponential filter smoothing
            _trydecomp = try to decompose with temperature trend or not
            _plotsInter = warning - many plots (True, False) plot intermediate analysis, 
            _plotResult = (True, False) plot final result, 
            _verbose = warning - many things (True, False), 
            _printStats = (True, False) print final statistics) 

        Output:
            _dataframe with pollutants added
            _metadata with statistics analysis
    '''
    
    dataframeResult = _dataframe.copy()
    numberSensors = len(_pollutantTuples)
    CorrParamsDict = dict()

    alpha_calData = getCalData('alphasense', _calibrationDataPath)
    
    for sensor in range(numberSensors):
        
        # Get Sensor 
        pollutant = _pollutantTuples[sensor][0]
        sensorID = _pollutantTuples[sensor][1]
        method = _pollutantTuples[sensor][2]
        if method == 'baseline':
            baselineType = _pollutantTuples[sensor][3]
        slot = _pollutantTuples[sensor][4]
        _deltas = _pollutantTuples[sensor][5]
        
        # Get Sensor data
        Sensitivity_1 = alpha_calData.loc[sensorID,'Sensitivity 1']
        Sensitivity_2 = alpha_calData.loc[sensorID,'Sensitivity 2']
        Target_1 = alpha_calData.loc[sensorID,'Target 1']
        Target_2 = alpha_calData.loc[sensorID,'Target 2']
        nWA = alpha_calData.loc[sensorID,'Zero Current']/alpha_calData.loc[sensorID,'Aux Zero Current']

        if not Target_1 == pollutant:
            print ('Sensor ID ({}) and pollutant type ({}) not matching'.format(Target_1, pollutant))
            return

        ## Retrieve alphasense ids from table (for the slot we are calcualting)
        for item in as_ids_table:
            if slot == item[0]:
                id_alphaW = item[1]
                id_alphaA = item[2]

        ## Retrieve alphasense name
        for currentSensorName in _currentSensorNames:        
            if id_alphaW == _currentSensorNames[currentSensorName]['id']:
                alphaW = _currentSensorNames[currentSensorName]['shortTitle']
            if id_alphaA == _currentSensorNames[currentSensorName]['id']:
                alphaA = _currentSensorNames[currentSensorName]['shortTitle']
       
        # Retrieve temperature and humidity names
        for item in th_ids_table:
            for currentSensorName in _currentSensorNames:      
                if item[1] == _currentSensorNames[currentSensorName]['id'] and _currentSensorNames[currentSensorName]['shortTitle'] in dataframeResult.columns:
                    temp = _currentSensorNames[currentSensorName]['shortTitle']
                    break
        
        for item in th_ids_table:
            for currentSensorName in _currentSensorNames:   
                if item[2] == _currentSensorNames[currentSensorName]['id'] and _currentSensorNames[currentSensorName]['shortTitle'] in dataframeResult.columns:
                    hum = _currentSensorNames[currentSensorName]['shortTitle']
                    break

        _listNames = (alphaW, alphaA, temp, hum)

        if pollutant == 'O3':
            # Check if NO2 is already present in the dataset
            if ('NO2'+ '_' + _append) in dataframeResult.columns:
                pollutant_column_2 = ('NO2' + '_' + _append)
            else:
                print ('Change tuple order to [(CO,sensorID_CO, ...), (NO2, sensorID_NO2, ...), (O3, sensorID_O3, ...)]')
                return
        
        # Get units for the pollutant in questions
        for pollutantItem in alphaUnitsFactorsLUT:
            
            if pollutant == pollutantItem[0]:
                factor_unit_1 = pollutantItem[1]
                factor_unit_2 = pollutantItem[2]

        ## Find min, max and range of days
        min_date_df, max_date_df, range_days = findDates(dataframeResult)
        print ('------------------------------------------------------------------')
        print ('Calculation of ' + '\033[1m{:10s}\033[0m'.format(pollutant))
        print ('Data Range from {} to {} with {} days'.format(min_date_df, max_date_df, range_days))
        print ('------------------------------------------------------------------')
        
        # Give name to pollutant column
        pollutant_column = (pollutant + '_' + _append) 
        
        if method == 'baseline':
            # Select baselined - baseliner depending on the baseline type
            if baselineType == 'single_temp':
                baseliner = temp
            elif baselineType == 'single_hum':
                baseliner = hum
            elif baselineType == 'single_aux':
                baseliner = alphaA
            print ('Using {} as baseliner'.format(baseliner))
            baselined = alphaW
            
            # Iterate over days
            for day in range(range_days):
            
                # Calculate overlap dates for that day
                min_date_ovl = max(_dataframe.index.min(), (min_date_df + pd.DateOffset(days=day) - pd.DateOffset(hours = _overlapHours)))
                max_date_ovl = min(_dataframe.index.max(), (min_date_ovl + pd.DateOffset(days=1) + pd.DateOffset(hours = _overlapHours + relativedelta.relativedelta(min_date_df + pd.DateOffset(days=day),min_date_ovl).hours)))
                
                # Calculate non overlap dates for that day
                min_date_novl = max(min_date_df, (min_date_df + pd.DateOffset(days=day)))
                max_date_novl = min(max_date_df, (min_date_novl + pd.DateOffset(days=1)))
            
                if _verbose:
                    print ('------------------------------------------------------------------')
                    print ('Calculating day {}, with range: {} \t to {}'.format(day, min_date_ovl, max_date_ovl))
                    print ('------------------------------------------------------------------')
                
                ## Trim dataframe to overlap dates
                dataframeTrim = dataframeResult[dataframeResult.index > min_date_ovl]
                dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_ovl]

                # Make 
                dataframeCheck = dataframeTrim.loc[:,(baselined, baseliner)].dropna()

                # Init stuff
                if day == 0:
                    # Init dict for CorrParams
                    CorrParams = list()
                 
                if dataframeCheck.empty:
                    CorrParamsTrim = dict()
                    CorrParamsTrim['0_valid'] = False
                    CorrParams.append(CorrParamsTrim)

                    if _verbose:
                        print ('No data between these dates' )

                else:
                    CorrParamsTrim = dict()

                    # CALCULATE THE BASELINE PER DAY
                    dataframeTrim[alphaW + '_BASELINE_' + _append], CorrParamsTrim = calculateBaselineDay(dataframeCheck, 'alphasense', _listNames, _deltas, _type_regress, _trydecomp, _plotsInter, _verbose)
                    
                    CorrParamsTrim['ratioAuxBase_avg']
                    # TRIM IT BACK TO NO-OVERLAP

                    dataframeTrim = dataframeTrim[dataframeTrim.index > min_date_novl]
                    dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_novl]

                    # CALCULATE ACTUAL POLLUTANT CONCENTRATION
                    if pollutant == 'CO': 
                        # Not recommended for CO
                        dataframeTrim[pollutant_column] = backgroundConc_CO + factor_unit_1*factorPCB*(dataframeTrim[alphaW] - dataframeTrim[alphaW + '_BASELINE_' + _append])/abs(Sensitivity_1)
                    elif pollutant == 'NO2':
                        dataframeTrim[pollutant_column] = backgroundConc_NO2 + factor_unit_1*factorPCB*(dataframeTrim[alphaW] - dataframeTrim[alphaW + '_BASELINE_' + _append])/abs(Sensitivity_1)
                        # dataframeTrim[pollutant_column] = backgroundConc_NO2 + factor_unit_1*factorPCB*(dataframeTrim[alphaW] - CorrParamsTrim['ratioAuxBase_avg']*dataframeTrim[alphaA])/abs(Sensitivity_1)
                    
                    elif pollutant == 'O3':
                        dataframeTrim[pollutant_column] = backgroundConc_OX + factor_unit_1*(factorPCB*(dataframeTrim[alphaW] - dataframeTrim[alphaW + '_BASELINE_' + _append]) - (dataframeTrim[pollutant_column_2])/factor_unit_2*abs(Sensitivity_2))/abs(Sensitivity_1)
                    
                    # ADD IT TO THE DATAFRAME
                    dataframeTrim[pollutant_column + '_FILTER'] = exponential_smoothing(dataframeTrim[pollutant_column].fillna(0), filterExpSmoothing)
                    dataframeResult = dataframeResult.combine_first(dataframeTrim)

                    if _refAvail:
                        ## Trim ref dataframe to no-overlap dates
                        dataframeTrimRef = _dataframeRef[_dataframeRef.index >= dataframeTrim.index.min()].fillna(0)
                        dataframeTrimRef = dataframeTrimRef[dataframeTrimRef.index <= dataframeTrim.index.max()].fillna(0)
                        
                        # Adapt dataframeTrim to be able to perform correlation
                        if dataframeTrimRef.index.min() > dataframeTrim.index.min():
                            dataframeTrim = dataframeTrim[dataframeTrim.index >= dataframeTrimRef.index.min()]
                        if dataframeTrimRef.index.max() < dataframeTrim.index.max():
                            dataframeTrim = dataframeTrim[dataframeTrim.index <= dataframeTrimRef.index.max()]

                        pollutant_ref = (pollutant + '_' + ref_append)

                        if pollutant_ref in dataframeTrimRef.columns and not dataframeTrimRef.empty:
                            CorrParamsTrim['r2_valueRef'] = r2_score(dataframeTrim[pollutant_column].fillna(0), dataframeTrimRef[pollutant_ref]) if not dataframeTrim.empty else np.nan
                            CorrParamsTrim['rmse'] = sqrt(mean_squared_error(dataframeTrim[pollutant_column].fillna(0), dataframeTrimRef[pollutant_ref])) if not dataframeTrim.empty else np.nan

                    else:
                        if _verbose:
                            print ('No Ref available')
                    
                    ## Get some metrics 
                    CorrParamsTrim['temp_avg'] = dataframeTrim[temp].mean() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['temp_stderr'] = dataframeTrim[temp].std() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['hum_avg'] = dataframeTrim[hum].mean() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['hum_stderr'] = dataframeTrim[hum].std() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['pollutant_avg'] = dataframeTrim[pollutant_column].mean() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['pollutant_std'] = dataframeTrim[pollutant_column].std() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['pollutant_min'] = dataframeTrim[pollutant_column].min() if not dataframeTrim.empty else np.nan
                    CorrParamsTrim['pollutant_max'] = dataframeTrim[pollutant_column].max() if not dataframeTrim.empty else np.nan

                    # if not dataframeTrim.empty:
                    #     days_with_data.append(day) 
                    CorrParams.append(CorrParamsTrim)
            
            CorrParamsDF = pd.DataFrame(CorrParams, index = [(min_date_df+ pd.DateOffset(days=day)).strftime('%Y-%m-%d') for day in range(range_days)])

            ## Find average ratio for hole dataset
            deltaAuxBas_avg = CorrParamsDF.loc[CorrParamsDF['0_valid'].fillna(False), 'deltaAuxBase_avg'].mean(skipna = True)
            deltaAuxBas_std = CorrParamsDF.loc[CorrParamsDF['0_valid'].fillna(False), 'deltaAuxBase_avg'].std(skipna = True)
            ratioAuxBas_avg = CorrParamsDF.loc[CorrParamsDF['0_valid'].fillna(False), 'ratioAuxBase_avg'].mean(skipna = True)
            ratioAuxBas_std = CorrParamsDF.loc[CorrParamsDF['0_valid'].fillna(False), 'ratioAuxBase_avg'].std(skipna = True)
                    
            # SHOW SOME METADATA FOR THE BASELINES FOUND
            if _printStats:
                        
                print ('------------------------')
                print (' Meta Data')
                print ('------------------------')
                display(CorrParamsDF)
                        
                print ('------------------------')
                print ('Average Delta between baseline and auxiliary electrode: {}, and ratio: {}'.format(deltaAuxBas_avg, ratioAuxBas_avg))
                print ('Std Dev of Delta between baseline and auxiliary electrode: {}, and ratio: {}'.format(deltaAuxBas_std, ratioAuxBas_std))
                print ('------------------------')
                    
        elif method == 'classic':
            
            ## CorrParams
            CorrParamsTrim = list()
            
            if pollutant == 'CO': 
                dataframeResult[pollutant_column] = factor_unit_1*factorPCB*(dataframeResult[alphaW] - nWA*dataframeResult[alphaA])/abs(Sensitivity_1)+backgroundConc_CO
            elif pollutant == 'NO2':
                dataframeResult[pollutant_column] = factor_unit_1*factorPCB*(dataframeResult[alphaW] - nWA*dataframeResult[alphaA])/abs(Sensitivity_1)+backgroundConc_NO2
            elif pollutant == 'O3':
                dataframeResult[pollutant_column] = factor_unit_1*(factorPCB*(dataframeResult[alphaW] - nWA*dataframeResult[alphaA]) - (dataframeResult[pollutant_column_2])/factor_unit_2*abs(Sensitivity_2))/abs(Sensitivity_1) + backgroundConc_OX
            
            dataframeResult[pollutant_column + '_FILTER'] = exponential_smoothing(dataframeResult[pollutant_column].fillna(0), filterExpSmoothing)

            ## Calculate stats day by day to avoid stationarity
            min_date_df, max_date_df, range_days = findDates(dataframeResult)
            # print 'Data Range from {} to {} with {} days'.format(min_date_df, max_date_df, range_days)
            
            for day in range(range_days):
                ## CorrParams
                CorrParamsTrim = {}
                
                # Calculate non overlap dates for that day
                min_date_novl = max(min_date_df, (min_date_df + pd.DateOffset(days=day)))
                max_date_novl = min(max_date_df, (min_date_novl + pd.DateOffset(days=1)))
                
                ## Trim dataframe to no-overlap dates
                dataframeTrim = dataframeResult[dataframeResult.index > min_date_novl].fillna(0)
                dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_novl].fillna(0)
                
                if _refAvail:
                    ## Trim ref dataframe to no-overlap dates
                    dataframeTrimRef = _dataframeRef[_dataframeRef.index >= dataframeTrim.index.min()].fillna(0)
                    dataframeTrimRef = dataframeTrimRef[dataframeTrimRef.index <= dataframeTrim.index.max()].fillna(0)
                    
                    # Adapt dataframeTrim to be able to perform correlation
                    if dataframeTrimRef.index.min() > dataframeTrim.index.min():
                        dataframeTrim = dataframeTrim[dataframeTrim.index >= dataframeTrimRef.index.min()]
                    if dataframeTrimRef.index.max() < dataframeTrim.index.max():
                        dataframeTrim = dataframeTrim[dataframeTrim.index <= dataframeTrimRef.index.max()]                    
                    pollutant_ref = pollutant + '_' + ref_append
                    if pollutant_ref in dataframeTrimRef.columns and not dataframeTrimRef.empty:
                        CorrParamsTrim['r2_valueRef'] = r2_score(dataframeTrim[pollutant_column], dataframeTrimRef[pollutant_ref]) if not dataframeTrim.empty else np.nan
                        CorrParamsTrim['rmse'] = sqrt(mean_squared_error(dataframeTrim[pollutant_column], dataframeTrimRef[pollutant_ref])) if not dataframeTrim.empty else np.nan

                else:
                    if _verbose:
                        print ('No ref Available')
                
                ## Get some metrics 
                CorrParamsTrim['temp_avg'] = dataframeTrim[temp].mean(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['temp_stderr'] = dataframeTrim[temp].std(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['hum_avg'] = dataframeTrim[hum].mean(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['hum_stderr'] = dataframeTrim[hum].std(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['pollutant_avg'] = dataframeTrim[pollutant_column].mean(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['pollutant_avg'] = dataframeTrim[pollutant_column + '_FILTER'].mean(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['pollutant_std'] = dataframeTrim[pollutant_column + '_FILTER'].std(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['pollutant_min'] = dataframeTrim[pollutant_column + '_FILTER'].min(skipna = True) if not dataframeTrim.empty else np.nan
                CorrParamsTrim['pollutant_max'] = dataframeTrim[pollutant_column + '_FILTER'].max(skipna = True) if not dataframeTrim.empty else np.nan


                if day == 0:
                    CorrParams = list()
                
                CorrParams.append(CorrParamsTrim)
            
            CorrParamsDF = pd.DataFrame(CorrParams, index = [(min_date_df+ pd.DateOffset(days=days)).strftime('%Y-%m-%d') for days in range(range_days)])
            
            # SHOW SOME METADATA FOR THE BASELINES FOUND
            if _printStats:
            
                print ('------------------------')
                print (' Meta Data')
                print ('------------------------')
                display(CorrParamsDF)
        
        # FILTER IT
        # dataframeResult[pollutant_column + '_filter'] = exponential_smoothing(dataframeResult[pollutant_column].fillna(0), filterExpSmoothing)
        
        ## RETRIEVE METADATA
        CorrParamsDict[pollutant] = CorrParamsDF
        
        ## TODO - Make the check for outliers and mark them out
        # CorrParamsDF['valid'] = deltaAuxBas_avg-deltaAuxBas_std <= CorrParamsDF['deltaAuxBas_avg'] <= deltaAuxBas_avg-deltaAuxBas_std
        # CorrParamsDF['valid'] = ratioAuxBas_avg-ratioAuxBas_std <= CorrParamsDF['ratioAuxBas_avg'] <= ratioAuxBas_avg-ratioAuxBas_std
        # 
        # print CorrParamsDF
        # 
        # deltaAuxBas_avg = CorrParamsDF.loc[CorrParamsDF['valid'].fillna(False), 'deltaAuxBas_avg'].mean(skipna = True)
        # deltaAuxBas_std = CorrParamsDF.loc[CorrParamsDF['valid'].fillna(False), 'deltaAuxBas_avg'].std(skipna = True)
        # ratioAuxBas_avg = CorrParamsDF.loc[CorrParamsDF['valid'].fillna(False), 'ratioAuxBas_avg'].mean(skipna = True)
        # ratioAuxBas_std = CorrParamsDF.loc[CorrParamsDF['valid'].fillna(False), 'ratioAuxBas_avg'].std(skipna = True)
        
        # PLOT THINGS IF REQUESTED
        if _plotResult:

            fig1 = tls.make_subplots(rows=4, cols=1, shared_xaxes=True, print_grid=False)
            
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaW], 'type': 'scatter', 'line': dict(width = 2), 'name': dataframeResult[alphaW].name}, 1, 1)
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaA], 'type': 'scatter', 'line': dict(width = 2), 'name': dataframeResult[alphaA].name}, 1, 1)
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaA] * nWA, 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': 'AuxCor Alphasense'}, 1, 1)
            if method == 'baseline':
                fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaW + '_BASELINE_' + _append], 'type': 'scatter', 'line': dict(width = 2), 'name': 'Baseline'}, 1, 1)
            
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[pollutant_column], 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': dataframeResult[pollutant_column].name}, 2, 1)
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[pollutant_column + '_FILTER'], 'type': 'scatter', 'name': (dataframeResult[pollutant_column + '_FILTER'].name)}, 2, 1)
            
            if _refAvail:
                # take the reference and check if it's available
                pollutant_ref = pollutant + '_' + ref_append
                if pollutant_ref in _dataframeRef.columns:
                    # If all good, plot it
                    fig1.append_trace({'x': _dataframeRef.index, 'y': _dataframeRef[pollutant_ref], 'type': 'scatter', 'name': _dataframeRef[pollutant_ref].name}, 2, 1)
                
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[temp], 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': dataframeResult[temp].name}, 3, 1)
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[hum], 'type': 'scatter', 'name': (dataframeResult[hum].name)}, 4, 1)
            
            fig1['layout'].update(height = 1500, 
                                  legend=dict(x=-.1, y=0.9), 
                                  xaxis=dict(title='Time'), 
                                  title = 'Baseline Correction for {}'.format(pollutant),
                                  yaxis1 = dict(title='Sensor Output - mV'), 
                                  yaxis2 = dict(title='Pollutant - ppm'),
                                  yaxis3 = dict(title='Temperature - degC'),
                                  yaxis4 = dict(title='Humidity - %'),
                                 )
                                   
            ply.offline.iplot(fig1)

    return dataframeResult, CorrParamsDict

def calculatePollutantsMICS(_dataframe, _pollutantTuples, _append, _refAvail, _dataframeRef, _deltas, _overlapHours = 0, _type_regress = 'best', _filterExpSmoothing = 0.2, _trydecomp = False, _plotsInter = False, _plotResult = True, _verbose = False, _printStats = False):
    '''
        Function to calculate mics pollutants with baseline technique
        Input:
            _dataframe: pandas dataframe from
            _pollutantTuples: list of tuples containing: 
                '[(_pollutant,
                    _sensorSN, 
                    calibration_method:
                        'baseline', 
                    baseline_type: 'baseline using temperature or humidity'
                        'single_temp'
                        'single_hum', 
                ...]'
            _append: suffix to the new channel name (pollutant + append)
            _refAvail: True or False if there is a reference available
            _dataframeRef: reference dataframe if available
            _deltas: for baseline correction method
            _overlapHours: number of hours to overlap over the day -> -_overlapHours+day:day+1+_overlapHours
            _type_regress = type of regression for baseline (best, exponential or linear)
            _filterExpSmoothing = alpha parameter for exponential filter smoothing
            _trydecomp = try to decompose with temperature trend or not
            _plotsInter = warning - many plots (True, False) plot intermediate analysis, 
            _plotResult = (True, False) plot final result, 
            _verbose = warning - many things (True, False), 
            _printStats = (True, False) print final statistics) 

        Output:
            _dataframe with pollutants added
            _metadata with statistics analysis
    '''
    
    dataframeResult = _dataframe.copy()
    numberSensors = len(_pollutantTuples)
    CorrParamsDict = dict()
    
    for sensor in range(numberSensors):
        
        # Get Sensor 
        pollutant = _pollutantTuples[sensor][0]
        sensorSN = _pollutantTuples[sensor][1]
        method = _pollutantTuples[sensor][2]
        
        if method == 'baseline':
            baselineType = _pollutantTuples[sensor][3]
        
        # Get Sensor data
        if pollutant == 'CO':
            indexPollutant = 1
        elif pollutant == 'NO2':
            indexPollutant = 2

        Sensitivity = mics_calData.loc[sensorSN,'Sensitivity ' +str(indexPollutant)]
        Target = mics_calData.loc[sensorSN,'Target ' + str(indexPollutant)]
        Zero_Air_Resistance = mics_calData.loc[sensorSN,'Zero Air Resistance ' + str(indexPollutant)]

        if not Target == pollutant:
            print ('Sensor ID ({}) and pollutant type ({}) not matching'.format(Target, pollutant))
            return

        mics_resist = CHANNEL_NAME(currentSensorNames, 'SENSOR_' + pollutant, '', '', 'BOARD_URBAN','kOhm')
        temp = CHANNEL_NAME(currentSensorNames, 'TEMPERATURE', '', '', 'BOARD_URBAN','C')
        hum = CHANNEL_NAME(currentSensorNames, 'HUMIDITY', '', '', 'BOARD_URBAN','%')
        
        _listNames = (mics_resist, temp, hum)
        
        # Get units for the pollutant in questions
        for pollutantItem in micsUnitsFactorsLUT:
            if pollutant == pollutantItem[0]:
                factor_unit = pollutantItem[1]

        ## Find min, max and range of days
        min_date_df, max_date_df, range_days = findDates(_dataframe)
        print ('------------------------------------------------------------------')
        print ('Calculation of ' + '\033[1m{:10s}\033[0m'.format(pollutant))
        print ('Data Range from {} to {} with {} days'.format(min_date_df, max_date_df, range_days))
        print ('------------------------------------------------------------------')
        
        # Give name to pollutant column
        pollutant_column = (pollutant + '_' + _append) 
        
        if method == 'baseline':

            # Select baselined - baseliner depending on the baseline type
            if baselineType == 'single_temp':
                baseliner = temp
            elif baselineType == 'single_rel_hum':
                baseliner = hum

            baselined = mics_resist
            
            # Iterate over days
            for day in range(range_days):
            
                # Calculate overlap dates for that day
                min_date_ovl = max(_dataframe.index.min(), (min_date_df + pd.DateOffset(days=day) - pd.DateOffset(hours = _overlapHours)))
                max_date_ovl = min(_dataframe.index.max(), (min_date_ovl + pd.DateOffset(days=1) + pd.DateOffset(hours = _overlapHours + relativedelta.relativedelta(min_date_df + pd.DateOffset(days=day),min_date_ovl).hours)))
                
                # Calculate non overlap dates for that day
                min_date_novl = max(min_date_df, (min_date_df + pd.DateOffset(days=day)))
                max_date_novl = min(max_date_df, (min_date_novl + pd.DateOffset(days=1)))
            
                if _verbose:
                    print ('------------------------------------------------------------------')
                    print ('Calculating day {}, with range: {} \t to {}'.format(day, min_date_ovl, max_date_ovl))
                    print ('------------------------------------------------------------------')
                
                ## Trim dataframe to overlap dates
                dataframeTrim = dataframeResult[dataframeResult.index > min_date_ovl]
                dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_ovl]
                
                # Init stuff
                if day == 0:
                    # Init list for CorrParams
                    CorrParams = list()
                 
                if dataframeTrim.empty:
                    if _verbose:
                        print ('No data between these dates')
                    
                    # Fill with nan if no data available (to avoid messing up the stats)
                    nanV =np.ones(10)
                    nanV.fill(np.nan)

                    CorrParams.append(tuple(nanV))
                    
                else:
                    
                    # CALCULATE THE BASELINE PER DAY
                    dataframeTrim[mics_resist + '_BASELINE_' + _append], CorrParamsTrim = calculateBaselineDay(dataframeTrim, 'mics', _listNames, baselined, baseliner, _deltas, _type_regress, _trydecomp, _plotsInter, _verbose)
                    
                    # TRIM IT BACK TO NO-OVERLAP
                    dataframeTrim = dataframeTrim[dataframeTrim.index > min_date_novl].fillna(0)
                    dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_novl].fillna(0)
                    
                    # CALCULATE ACTUAL POLLUTANT CONCENTRATION
                    pollutant_wo_background = factor_unit*(dataframeTrim[mics_resist] - dataframeTrim[mics_resist + '_BASELINE_' + _append] - Zero_Air_Resistance)/Sensitivity
                    
                    if pollutant == 'CO': 
                        dataframeTrim[pollutant_column] = backgroundConc_CO + pollutant_wo_background
                    elif pollutant == 'NO2':
                        dataframeTrim[pollutant_column] = backgroundConc_NO2 + pollutant_wo_background

                    # ADD IT TO THE DATAFRAME
                    dataframeResult = dataframeResult.combine_first(dataframeTrim)
                    
                    if _refAvail:
                        pollutant_ref = pollutant + '_' + ref_append
                        ## Trim ref dataframe to no-overlap dates
                        dataframeTrimRef = _dataframeRef[_dataframeRef.index >= dataframeTrim.index.min()].fillna(0)
                        dataframeTrimRef = dataframeTrimRef[dataframeTrimRef.index <= dataframeTrim.index.max()].fillna(0)
                        
                        # Adapt dataframeTrim to be able to perform correlation
                        if dataframeTrimRef.index.min() > dataframeTrim.index.min():
                            dataframeTrim = dataframeTrim[dataframeTrim.index >= dataframeTrimRef.index.min()]
                        if dataframeTrimRef.index.max() < dataframeTrim.index.max():
                            dataframeTrim = dataframeTrim[dataframeTrim.index <= dataframeTrimRef.index.max()]
                        
                        if pollutant_ref in dataframeTrimRef.columns and not dataframeTrimRef.empty:
                            slopeRef, interceptRef, r_valueRef, p_valueRef, std_errRef = linregress(np.transpose(dataframeTrim[pollutant_column]), np.transpose(dataframeTrimRef[pollutant_ref]))
                        else:
                            r_valueRef = np.nan
                    else:
                        r_valueRef = np.nan
                        # print 'No Ref available'
                    
                    ## Get some metrics
                    temp_avg = dataframeTrim[temp].mean(skipna = True)
                    temp_stderr = dataframeTrim[temp].std(skipna = True)
                    hum_avg = dataframeTrim[hum].mean(skipna = True)
                    hum_stderr = dataframeTrim[hum].std(skipna = True)
                    pollutant_avg = dataframeTrim[pollutant_column].mean(skipna = True)
                    
                    tempCorrParams = list(CorrParamsTrim)
                    tempCorrParams.insert(0,r_valueRef**2)
                    tempCorrParams.insert(1,pollutant_avg)
                    tempCorrParams.insert(1,hum_stderr)
                    tempCorrParams.insert(1,hum_avg)
                    tempCorrParams.insert(1,temp_stderr)
                    tempCorrParams.insert(1,temp_avg)                    
                    CorrParamsTrim = tuple(tempCorrParams)
                    CorrParams.append(CorrParamsTrim)
            
            ## Add relevant metadata for this method
            labelsCP = ['r_valueRef',
                        'avg_temp',
                        'stderr_temp',
                        'avg_hum',
                        'stderr_hum',
                        'avg_pollutant',
                        'indexMax']
            
            CorrParamsDF = pd.DataFrame(CorrParams, columns = labelsCP, index = [(min_date_df+ pd.DateOffset(days=days)).strftime('%Y-%m-%d') for days in range(range_days)])
                   
            # SHOW SOME METADATA FOR THE BASELINES FOUND
            if _printStats:
                        
                print ('------------------------')
                print (' Meta Data')
                print ('------------------------')
                display(CorrParamsDF)
                        
        elif method == 'classic':
            print('Nothing to see here')
            # Nothing
        
        # FILTER IT
        # dataframeResult[pollutant_column + '_filter'] = exponential_smoothing(dataframeResult[pollutant_column].fillna(0), filterExpSmoothing)
        
        ## RETRIEVE METADATA
        CorrParamsDict[pollutant] = CorrParamsDF
        
        # PLOT THINGS IF REQUESTED
        if _plotResult:
            
            fig1 = tls.make_subplots(rows=4, cols=1, shared_xaxes=True, print_grid=False)
            
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[mics_resist], 'type': 'scatter', 'line': dict(width = 2), 'name': dataframeResult[mics_resist].name}, 1, 1)
            if method == 'baseline':
                fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[mics_resist + '_BASELINE_' + _append], 'type': 'scatter', 'line': dict(width = 2), 'name': 'Baseline'}, 1, 1)
            
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[pollutant_column], 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': dataframeResult[pollutant_column].name}, 2, 1)
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[pollutant_column + '_FILTER'], 'type': 'scatter', 'name': (dataframeResult[pollutant_column + '_FILTER'].name)}, 2, 1)
            
            if _refAvail:
                # take the reference and check if it's available
                if pollutant_ref in _dataframeRef.columns:
                    # If all good, plot it
                    fig1.append_trace({'x': _dataframeRef.index, 'y': _dataframeRef[pollutant_ref], 'type': 'scatter', 'name': _dataframeRef[pollutant_ref].name}, 2, 1)
                
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[temp], 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': dataframeResult[temp].name}, 3, 1)
            fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[hum], 'type': 'scatter', 'name': (dataframeResult[hum].name)}, 4, 1)
            
            fig1['layout'].update(height = 1500, 
                                  legend=dict(x=-.1, y=0.9), 
                                  xaxis=dict(title='Time'), 
                                  title = 'Baseline Correction for {}'.format(pollutant),
                                  yaxis1 = dict(title='Sensor Output - mV'), 
                                  yaxis2 = dict(title='Pollutant - ppm'),
                                  yaxis3 = dict(title='Temperature - degC'),
                                  yaxis4 = dict(title='Humidity - %'),
                                 )
                                   
            ply.offline.iplot(fig1)

    return dataframeResult, CorrParamsDict

