from numpy import ones, transpose, log
from scipy.stats.stats import linregress
from scipy.sparse import (diags, spdiags)
from scipy.sparse.linalg import spsolve
from pandas import date_range
from numpy import min as npmin
from numpy import max as npmax
from numpy import abs as npabs
from numpy import argmax, argmin, arange, exp
from scdata.utils import std_out
from scdata._config import config
from math import isnan
from .formulae import exp_f
import matplotlib.pyplot as plt
from re import search

def find_min_max(min_max, iterable = list()):
    """
    Returns the value and index of maximum in the list
    Parameters
    ----------
        min_max: 'string'
            whether to find the 'min' or the 'max'
        iterable: list
            list to obtain maximum value
    Returns
    -------
        Value and index of maximum in the list
    """   

    if min_max == 'max':
        value = npmax(iterable)
        index = argmax(iterable)
    elif min_max == 'min':
        value = npmin(iterable)
        index = argmin(iterable)
    else: 
        value, index = None, None

    return value, index

def get_delta_baseline(series, **kwargs):
    """
    Baseline based on deltas method
    Parameters
    ----------
        series: pd.series
            The timeseries to be baselined
        delta: int
            The delta for getting the minimum based baseline
        btype: 'string' (optional)
            'min'
            If is a 'min' or a 'max' baseline
    Returns
    -------
        Series containing baselined values
    """

    if 'delta' in kwargs: delta = kwargs['delta']
    else: return None

    # if 'resample' in kwargs: resample = kwargs['resample']
    # else: resample = '1Min'

    if 'btype' in kwargs: btype = kwargs['btype']
    else: btype = 'min'

    if delta == 0: std_out(f'Not valid delta = {delta}', 'ERROR'); return None
    
    result = series.copy()
    # result = result.resample(resample).mean()

    pdates = date_range(start = result.index[0], end = result.index[-1], freq = f'{delta}Min')

    for pos in range(0, len(pdates)-1):
        chunk = series[pdates[pos]:pdates[pos+1]]
        
        if len(chunk.values) == 0: result[pdates[pos]:pdates[pos+1]] = 0
        else: 
            if btype == 'min': result[pdates[pos]:pdates[pos+1]] = min(chunk.values)
            elif btype == 'max': result[pdates[pos]:pdates[pos+1]] = max(chunk.values)
    
    return result

def get_als_baseline(series, lambd = 1e5, p = 0.01, n_iter=10):

    L = len(series)
    D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = ones(L)

    for i in range(n_iter):
        W = spdiags(w, 0, L, L)
        Z = W + lambd * D.dot(D.transpose())
        z = spsolve(Z, w*series)
        w = p * (series > z) + (1-p) * (series < z)
    
    return z

# TODO DOCUMENT
def baseline_calc(dataframe, **kwargs):

    '''
    reg_type
    baseline_type
      if als  
        lambdas
        p
      if deltas  
        esample: int (optional)
            '1Min'
            Frequency at which the delta is based on, and therefore to resample to
        deltas
    '''

    if 'reg_type' not in kwargs: reg_type = 'best'
    else: reg_type = kwargs['reg_type']

    if 'baseline_type' not in kwargs: baseline_type = 'deltas'
    else: baseline_type = kwargs['baseline_type']

    pearsons =[]
    target_name = dataframe.iloc[:,0].name; std_out ('Target: ', target_name)
    baseline_name = dataframe.iloc[:,1].name; std_out ('Baseline: ', baseline_name)

    result = dataframe.copy()
    result.dropna(axis = 0, inplace=True)

    if result.empty: return None

    if config._intermediate_plots and config._plot_out_level == 'DEBUG': 
        fig, ax = plt.subplots(figsize=(12,8))

    if baseline_type == 'deltas':

        if 'deltas' not in kwargs: n_deltas = config._baseline_deltas
        else: n_deltas = kwargs['deltas']
        
        if 'resample' not in kwargs: resample = '1Min'
        else: resample = kwargs['resample']

        result = result.resample(resample).mean()
        
        l_iter = n_deltas

        for delta in n_deltas:

            name_delta = target_name +'_' +str(delta)

            result[name_delta] = get_delta_baseline(result.loc[:,target_name], delta = delta)

            # Try to resample to improve correlation of baseline and target
            off_base = int(search(r'\d+', resample).group())
            off_alias = ''.join(i for i in resample if not i.isdigit())

            target_resampled = result.loc[:,name_delta].resample(f'{delta*off_base}{off_alias}').mean().values
            baseline_resampled = result.loc[:,baseline_name].resample(f'{delta*off_base}{off_alias}').mean().values

            if config._intermediate_plots and config._plot_out_level == 'DEBUG': 
                ax.plot(result.index, result[name_delta], label = name_delta)

            _, _, r_value, _, _ = linregress(transpose(target_resampled), transpose(baseline_resampled))
            pearsons.append(r_value)

    elif baseline_type == 'als':

        if 'lambdas' not in kwargs: lambdas = config._baseline_als_lambdas
        else: lambdas = kwargs['lambdas']

        if 'p' not in kwargs: p = 0.01
        else: p = kwargs['p']

        l_iter = lambdas

        for lambd in lambdas:

            name_lambda = name +'_' +str(lambd)
            result[name_lambda] = get_als_baseline(result.loc[:,target_name], lambd, p)

            if config._intermediate_plots and config._plot_out_level == 'DEBUG': 
                ax.plot(result.index, result[name_lambda], label = name_lambda)

            _, _, r_value, _, _ = linregress(transpose(result[name_lambda]), transpose(result.loc[:,baseline_name].values))
            pearsons.append(r_value)

    if config._intermediate_plots and config._plot_out_level == 'DEBUG':
        plt.show()
        ax.plot(result.index, result.loc[:,target_name], label = target_name)
        ax.plot(result.index, result.loc[:,baseline_name], label = baseline_name)
        
        ax.axis('tight')
        ax.legend(loc='best')
        ax.set_xlabel('Date')
        ax.set_ylabel('Baselines')
        ax.grid(True)
    
        plt.show()

    ## Find Max in the pearsons - correlation can be negative, so use absolute of the pearson
    _, ind_max = find_min_max('max', npabs(pearsons))
    # std_out(f'Max index in pearsons: {ind_max}')
    result.dropna(axis = 0, inplace=True)

    if reg_type == 'linear':
        
        ## Fit with y = A + Bx
        slope, intercept, r_value, p_value, std_err = linregress(transpose(result.loc[:,baseline_name].values), result[(target_name + f'_{l_iter[ind_max]}')])
        baseline = intercept + slope*result.loc[:,baseline_name].values
        # print (r_value)
    
    elif reg_type == 'exponential':
        
        ## Fit with y = Ae^(Bx) -> logy = logA + Bx
        logy = log(result[(target_name + f'_{l_iter[ind_max]}')])
        slope, intercept, r_value, p_value, std_err = linregress(transpose(result.loc[:,baseline_name].values), logy)
        baseline = exp_f(transpose(result.loc[:,baseline_name].values), exp(intercept), slope, 0)
        # print (r_value)
    
    elif reg_type == 'best':
        
        ## Find linear r_value
        slope_lin, intercept_lin, r_value_lin, p_value_lin, std_err_lin = linregress(transpose(result.loc[:, baseline_name].values), result[(target_name + f'_{l_iter[ind_max]}')])
        
        ## Find Exponential r_value
        logy = log(result[(target_name + f'_{l_iter[ind_max]}')])
        slope_exp, intercept_exp, r_value_exp, p_value_exp, std_err_exp = linregress(transpose(result.loc[:, baseline_name].values), logy)

        ## Pick which one is best
        if ((not isnan(r_value_exp)) and (not isnan(r_value_lin))):

            if r_value_lin > r_value_exp:
                baseline = intercept_lin + slope_lin*result.loc[:,baseline_name].values
            else:
                baseline = exp_f(transpose(result.loc[:,baseline_name].values), exp(intercept_exp), slope_exp, 0)
        
        elif not isnan(r_value_lin):
            
            baseline = intercept_lin + slope_lin*result.loc[:,baseline_name].values
        
        elif not isnan(r_value_exp):
            
            baseline = exp_f(transpose(result.loc[:,baseline_name].values), exp(intercept_exp), slope_exp, 0)
        else:
            std_out('Exponential and linear regression are nan', 'ERROR')
    
    # Avoid baselines higher than the target
    result[target_name + '_baseline_raw'] = baseline
    result[target_name + '_baseline'] = result[[(target_name + '_' + 'baseline_raw'), target_name]].min(axis=1)
    
    if config._intermediate_plots and config._plot_out_level == 'DEBUG':
        with plt.style.context('seaborn-white'):
            fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
            
            ax1.plot(result.loc[:, baseline_name].values, result[(target_name + f'_{l_iter[ind_max]}')], label = 'Baseline ' + str(l_iter[ind_max]), linewidth=0, marker='o')
            ax1.plot(result.loc[:, baseline_name].values, result[(target_name + '_baseline')] , label = 'Regressed value', linewidth=0, marker='o')
            legend = ax1.legend(loc='best')
            ax1.set_xlabel(baseline_name)
            ax1.set_ylabel('Regression values')
            ax1.grid(True)
            
            lns1 = ax2.plot(result.index, result.loc[:, target_name], label = "Target", linestyle=':', linewidth=1, marker=None)
            #[ax2.plot(result.index, result[(name +'_' +str(delta))].values, label="Delta {}".format(delta), marker=None,  linestyle='-', linewidth=1) for delta in _numberDeltas]
            lns2 = ax2.plot(result.index, result[target_name + '_' + 'baseline'], label='Baseline', marker = None)

            ax2.axis('tight')
            ax2.set_title("Baseline Extraction")
            ax2.grid(True)
            
            ax22 = ax2.twinx()
            lns22 = ax22.plot(result.index, result.loc[:, baseline_name].values, color = 'red', label = baseline_name, linestyle='-', linewidth=1, marker=None)
            ax22.set_ylabel(result.loc[:, baseline_name].name, color = 'red')
            ax22.set_ylim(ax2.get_ylim())
            ax22.tick_params(axis='y', labelcolor='red')

            lns = lns1+lns2+lns22
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc='best')
            
            fig2, ax3 = plt.subplots(figsize=(12,8)) # two axes on figure
            ax3.plot(l_iter, pearsons)

            if baseline_type == 'deltas':
                ax3.set_title("R2 vs. Delta")
                ax3.set_xlabel('Delta')

            elif baseline_type == 'als':
                ax3.set_title("R2 vs. λ")
                ax3.set_xlabel('λ')
                ax3.set_xscale('log')

            ax3.axis('tight')
            ax3.set_ylabel('R2')
            ax3.grid(True)
            plt.show()

    return result[target_name + '_' + 'baseline']