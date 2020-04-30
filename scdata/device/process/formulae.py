from numpy import exp, log, transpose
from scipy.stats.stats import linregress

# TODO REVIEW
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
    vap_eq = (1.0007 + 3.46*1e-6*press)*6.1121*exp(17.502*temp/(240.97+temp))

    abs_humidity = rel_h * vap_eq

    return abs_humidity

def exp_f(x, A, B, C):
    """
    Returns the result of the formula: y = A*e^(Bx) + C
    Parameters
    ----------
        x: pd.Series
            Variable x in the formula
        A: float
            A parameter in the formula
        B: float
            B parameter in the formula
        C: float
            C parameter in the formula
    Returns
    -------
        pandas series
    """    
    return A * exp(B * x) + C

def fit_exp_f(y, x):
    """
    Returns parameters A and B that would fit an exponential
    function of y = A*e^(Bx)
    Parameters
    ----------
        y: pd.Series
            Variable y in the formula    
        x: pd.Series
            Variable x in the formula
    Returns
    -------
        Parameters A and B
    """    

    ## Fit with y = Ae^(Bx) -> logy = logA + Bx
    # Returns A and B of a function as: y = A*e^(Bx)
    B, logA, r_value, p_value, std_err = linregress(transpose(x.values), log(y))
    
    return exp(logA), B