import statsmodels.formula.api as smform
import statsmodels.api as smapi
import statsmodels.graphics as smgraph
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.stattools import adfuller

import numpy as np
import pandas as pd

from math import sqrt

## Metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error#, mean_squared_log_error

# Plots
import matplotlib.pyplot as plot
import seaborn as sns
# Others
from src.models.formula_utils import exponential_smoothing
from src.data.signal_utils import metrics

def tfuller_plot(_x, name = '', lags=None, figsize=(12, 7), lags_diff = 1):
    
    if lags_diff > 0:
        _x = _x - _x.shift(lags_diff)
        
    _x = _x.dropna()
    
    ad_fuller_result = adfuller(_x)
    adf = ad_fuller_result[0]
    pvalue = ad_fuller_result[1]
    usedlag = ad_fuller_result[2]
    nobs = ad_fuller_result[3]
    print ('{}:'.format(name))
    print ('\tADF- Statistic: %.5f \tpvalue: %.5f \tUsed Lag: % d \tnobs: % d ' % (adf, pvalue, usedlag, nobs))

    with plot.style.context('seaborn-white'):    
        fig = plot.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plot.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plot.subplot2grid(layout, (1, 0))
        pacf_ax = plot.subplot2grid(layout, (1, 1))
        
        _x.plot(ax=ts_ax)
        ts_ax.set_ylabel(name)
        ts_ax.set_title('Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(pvalue))
        smgraph.tsaplots.plot_acf(_x, lags=lags, ax=acf_ax)
        smgraph.tsaplots.plot_pacf(_x, lags=lags, ax=pacf_ax)
        plot.tight_layout()

def prep_data_OLS(dataframeModel, tuple_features, ratio_train):
    '''
        Prepare Dataframe for Ordinary Linear Regression with StatsModels.
        Input:
            dataframeModel: Dataframe containing the data to be treated
            tuple_features: tuple containing features with [TERM, NAME, DEVICE]
            ratio_train: n_points_train/n_points_test+train
        Output:
            It returns two dataframes (train, test) with the columns named 
            as in TERM, with an additional constant value with name 'const', 
            for the independent term of the linear regression
    '''

    # Train Dataframe
    total_len = len(dataframeModel.index)
    n_train_periods = int(round(total_len*ratio_train))
    
    # Rename to formula
    for item in tuple_features:
        dataframeModel.rename(columns={'_'.join([item[1],item[2]]): item[0]}, inplace=True)
    
    dataframeTrain = dataframeModel.iloc[:n_train_periods,:]
    dataframeTrain['CONST'] = 1.0

    # Test Dataframe
    if ratio_train < 1:
        dataframeTest = dataframeModel.iloc[n_train_periods:,:]
        dataframeTest['CONST'] = 1.0
        
        return dataframeTrain, dataframeTest, n_train_periods

    return dataframeTrain, total_len

def fit_model_OLS(formula_expression, dataTrain, printSummary = True):
    '''
        
    '''
    model = smform.ols(formula = formula_expression, data = dataTrain).fit()
    
    if printSummary:
    	print(model.summary())

    return model

def plot_OLS_coeffs(model):
    """
        Plots sorted coefficient values of the model
    """
    _data =  [x for x in model.params]
    _columns = model.params.index
    _coefs = pd.DataFrame(_data, _columns, dtype = 'float')
    
    _coefs.columns = ["coef"]
    _coefs["abs"] = _coefs.coef.apply(np.abs)
    _coefs = _coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    figure = plot.figure(figsize=(15, 7))
    _coefs.coef.plot(kind='bar')
    plot.grid(True, axis='y')
    plot.hlines(y=0, xmin=0, xmax=len(_coefs), linestyles='dashed')
    plot.title('Linear Regression Coefficients')

def predict_OLS(model, data, plotResult = True, plotAnomalies = True, train_test = 'test'):    

    try:
        reference = data['REF']
        ref_avail = True
    except:
        # Do nothin
        ref_avail = False
        print ('No reference available')

    ## Predict Results
    if train_test == 'train':

        predictionTrain = model.predict(data)
        
        ## Get confidence intervals
        # For training
        st, summary_train, ss2 = summary_table(model, alpha=0.05)
        
        train_mean_se  = summary_train[:, 3]
        train_mean_ci_low, train_mean_ci_upp = summary_train[:, 4:6].T
        train_ci_low, train_ci_upp = summary_train[:, 6:8].T

        if plotResult:
            # Plot the stuff
            fig = plot.figure(figsize=(15,10))
            # Actual data
            plot.plot(data.index, reference, 'r', label = 'Reference Train', alpha = 0.3)
        
            # Fitted Values for Training
            plot.plot(data.index, predictionTrain, 'r', label = 'Prediction Train')
            plot.plot(data.index, train_ci_low, 'k--', lw=0.7, alpha = 0.5)
            plot.plot(data.index, train_ci_upp, 'k--', lw=0.7, alpha = 0.5)
            plot.fill_between(data.index, train_ci_low, train_ci_upp, alpha = 0.05 )
            plot.plot(data.index, train_mean_ci_low, 'r--', lw=0.7, alpha = 0.6)
            plot.plot(data.index, train_mean_ci_upp, 'r--', lw=0.7, alpha = 0.6)
            plot.fill_between(data.index, train_mean_ci_low, train_mean_ci_upp, alpha = 0.05 )
            plot.grid(True)

        if ref_avail:
            # Put train into pd dataframe
            dataFrameTrain = pd.DataFrame(data = {'reference': reference, 'prediction': predictionTrain.values}, 
                              index = data.index)
            return dataFrameTrain
        else:
            return predictionTrain

    elif train_test == 'test':

        predictionTest = model.get_prediction(data)

        ## Get confidence intervals
        # For test
        summary_test = predictionTest.summary_frame(alpha=0.05)
        test_mean = summary_test.loc[:, 'mean'].values
        test_mean_ci_low = summary_test.loc[:, 'mean_ci_lower'].values
        test_mean_ci_upp = summary_test.loc[:, 'mean_ci_upper'].values
        test_ci_low = summary_test.loc[:, 'obs_ci_lower'].values
        test_ci_upp = summary_test.loc[:, 'obs_ci_upper'].values

        if plotResult:
            # Plot the stuff
            fig = plot.figure(figsize=(15,10))
            # Fitted Values for Test
            plot.plot(data.index, reference, 'b', label = 'Reference Test', alpha = 0.3)

            plot.plot(data.index, test_mean, 'b', label = 'Prediction Test')
            plot.plot(data.index, test_ci_low, 'k--', lw=0.7, alpha = 0.5)
            plot.plot(data.index, test_ci_upp, 'k--', lw=0.7, alpha = 0.5)
            plot.fill_between(data.index, test_ci_low, test_ci_upp, alpha = 0.05 )
            plot.plot(data.index, test_mean_ci_low, 'r--', lw=0.7, alpha = 0.6)
            plot.plot(data.index, test_mean_ci_upp, 'r--', lw=0.7, alpha = 0.6)
            plot.fill_between(data.index, test_mean_ci_low, test_mean_ci_upp, alpha = 0.05 )
        
            plot.title('Linear Regression Results')
            plot.grid(True)
            plot.ylabel('Reference/Prediction (-)')
            plot.xlabel('Date (-)')
            plot.legend(loc='best')
            plot.show()

        if ref_avail:
            # Put test into pd dataframe
            dataFrameTest = pd.DataFrame(data = {'reference': reference, 'prediction': test_mean}, 
                              index = data.index)
            return dataFrameTest
        else:
            return test_mean

def model_R_plots(model, dataTrain, dataTest):
    
    ## Calculations required for some of the plots:
    # fitted values (need a constant term for intercept)
    model_fitted_y = model.fittedvalues
    # model residuals
    model_residuals = model.resid
    # normalized residuals
    model_norm_residuals = model.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model.get_influence().cooks_distance[0]

    ## Residual plot
    height = 6
    width = 8
    
    plot_lm_1 = plot.figure(1)
    plot_lm_1.set_figheight(height)
    plot_lm_1.set_figwidth(width)
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'REF', data=dataTrain,
                                      lowess=True,
                                      scatter_kws={'alpha': 0.5},
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')
    
    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_residuals[i]));

    QQ = smapi.ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    
    plot_lm_2.set_figheight(height)
    plot_lm_2.set_figwidth(width)
    
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
    
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]

    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.axes[0].annotate(i, 
                                   xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                       model_norm_residuals[i]));
    
    plot_lm_3 = plot.figure(3)
    plot_lm_3.set_figheight(height)
    plot_lm_3.set_figwidth(width)

    plot.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
    
    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    
    for i in abs_norm_resid_top_3:
        plot_lm_3.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_norm_residuals_abs_sqrt[i]));
    plot_lm_4 = plot.figure(4)
    plot_lm_4.set_figheight(height)
    plot_lm_4.set_figwidth(width)

    plot.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    plot_lm_4.axes[0].set_xlim(0, 0.20)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
    
    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i, 
                                   xy=(model_leverage[i], 
                                       model_norm_residuals[i]))

    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plot.plot(x, y, label=label, lw=1, ls='--', color='red')
    
    p = len(model.params) # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50), 
          'Cook\'s distance') # 0.5 line
    
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50)) # 1 line
    
    plot.legend(loc='upper right');

    # Model residuals
    tfuller_plot(model_residuals, name = 'Residuals', lags=60, lags_diff = 0)
