import numpy as np
import math
import pandas as pd

def exponential_smoothing(series, alpha):
    '''
        Input:
            series - dataset with timestamps
            alpha - float [0.0, 1.0], smoothing parameter
        Output: 
            smoothed series
    '''
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def SMOOTH(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def AD_FORMULA(WE, AE, SensorType, I0WE, I0AE, SENSITIVITY1, SENSITIVITY2, AUX, UNIT1, UNIT2):
    
    FACTORWE = 6.39
    FACTORAE = 6.35
    
    # import matplotlib.pyplot as plt
    # plt.plot(FACTORWE)
    # plt.plot(FACTORAE)

    if SensorType == 1 or 2:
        # CO OR NO2
        # CO: BOARD 5 AD_FORMULA(A, B, 1, -69.4, -18.6, 493.1, 0, C, "ppm", "")
        # NO2: BOARD U AD_FORMULA(A, B, 1, 31.5, 17.7, -383.7, 0, C, "ppb", "")
        # NO2: BOARD 5 AD_FORMULA(A, B, 1, 24.0, 14.2, -385.9, 0, C, "ppb", "")
        result = (FACTORWE*WE - I0WE/I0AE*(FACTORAE*AE))/abs(SENSITIVITY1)
        
        if UNIT1 == "ppb":
            result = result*1000
    
    if SensorType== 3:
        # O3
        # BOARD U AD_FORMULA(A, B, 3, 1, 23.65, 18.92, -421.58, -471.6497, C, "ppb", "ppb")
        # BOARD 5 AD_FORMULA(A, B, 3, 1, 23.33, 19.86, -433.12, -506.96, C, "ppb", "ppb")
    
        if UNIT2 =="ppb":
            result = (FACTORWE*WE - I0WE/I0AE*(FACTORAE*AE) - AUX*SENSITIVITY2/1000)/SENSITIVITY1
        if UNIT2 =="ppm":
            result = (FACTORWE*WE - I0WE/I0AE*(FACTORAE*AE) - AUX*SENSITIVITY2)/SENSITIVITY1
        if UNIT1 == "ppb":
            result = result*1000

    return result

def LINE_COEFF(X1,X2,Y1,Y2):
    a = float(Y2-Y1)/(X2-X1)
    b = Y1-a*X1
    return a,b

def LINE(x,a,b):
    y = [i*a+b for i in x]
    return y

def MICS_FORMULA(Sensor1Type, Sensor1, Sensor2Type, Sensor2, Sensor3Type, Sensor3, Intercept, B, C, D, E, F, G):
    SensorType = [Sensor1Type, Sensor2Type, Sensor3Type]
    Sensor = [Sensor1, Sensor2, Sensor3]
    Sens = Sensor
    for i in range(3):
        if SensorType[i] == "Inverse":
            Sens[i] = 1/Sensor[i]
    
    result =  Intercept + B*Sens[0] +C*Sens[0]*Sens[0] + D*Sens[1] + E*Sens[1]*Sens[1] + F*Sens[2] + G*Sens[2]*Sens[2]

    # MICS_Formula("Inverse", A, "Direct", B, "Direct", C, -1.615e-01 , 210.08064516, -10236.73257, 0, 0, 0, 0)
    return result

def ABS_HUM(temperature, rel_humidity, pressure):
    '''
        Calculate Absolute humidity based on vapour equilibrium:
        Input: 
            Temperature: in degC
            Rel_humidity: in %
            Pressure: in mbar
        Output:
            Absolute_humidity: in mg/m3?
    '''
    # Vapour equilibrium: 
    # _Temp is temperature in degC, _Press is absolute pressure in mbar
    vap_eq = (1.0007 + 3.46*1e-6*pressure)*6.1121*np.exp(17.502*temperature/(240.97+temperature))

    abs_humidity = rel_humidity * vap_eq

    return abs_humidity

def maxer(y, val):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<=val and not y[i] == np.nan): result[i] = val
        elif (y[i]>val and not y[i] == np.nan): result[i] = y[i]
        elif (math.isnan(y[i])): result[i] = np.nan
    return result

def maxer_hist(y, val, hist):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<=val and not y[i] == np.nan): result[i] = hist
        elif (y[i]>val and not y[i] == np.nan): result[i] = y[i]
        elif (math.isnan(y[i])): result[i] = np.nan
    return result

def miner(y, val):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<=val and not y[i] == np.nan): result[i] = y[i]
        elif (y[i]>val and not y[i] == np.nan): result[i] = val
        elif (math.isnan(y[i])): result[i] = np.nan
    return result

def greater(y, val):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<=val and not y[i] == np.nan): result[i] = False
        elif (y[i]>val and not y[i] == np.nan): result [i] = True
        elif (math.isnan(y[i])): result[i] = result[i-1]   
    return result

def greaterequal(y, val):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<val and not y[i] == np.nan): result[i] = False
        elif (y[i]>=val and not y[i] == np.nan): result [i] = True
        elif (math.isnan(y[i])): result[i] = result[i-1]
    return result

def lower (y, val):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<val and not y[i] == np.nan): result[i] = True
        elif (y[i]>=val and not y[i] == np.nan): result [i] = False
        elif (math.isnan(y[i])): result[i] = result[i-1]   
    return result

def lowerequal(y, val):
    result = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i]<=val and not y[i] == np.nan): result[i] = True
        elif (y[i]>val and not y[i] == np.nan): result [i] = False
        elif (math.isnan(y[i])): result[i] = result[i-1] 
    return result
            
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
    return series.diff(periods = -window)*60/series.index.to_series().diff(periods = -window).dt.total_seconds()

def exponential_func(x, a, b, c):
     return a * np.exp(b * x) + c

def evaluate(predictions, original):
    errors = abs(predictions - original)
    max_error = max(errors)
    rerror = np.maximum(np.minimum(np.divide(errors, original),1),-1)
    
    mape = 100 * np.mean(rerror)
    accuracy = 100 - mape
    
    return max_error, accuracy