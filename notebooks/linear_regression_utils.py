import statsmodels.formula.api as smform
import statsmodels.api as smapi
import statsmodels.graphics as smgraph
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.stattools import adfuller
import numpy as np

from math import sqrt

## Metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error#, mean_squared_log_error

# Plots
import matplotlib.pyplot as plot
import seaborn as sns
# %matplotlib inline
# Others
from formula_utils import exponential_smoothing

def prepData(dataframeModel, tuple_features, min_date, max_date, ratio_train, filter_data = False, alpha_filter = 0.2):

    ## Prepare Data Frame
    dataframeModel = dataframeModel[dataframeModel.index > min_date]
    dataframeModel = dataframeModel[dataframeModel.index < max_date]
    # Train Dataframe
    total_len = len(dataframeModel.index)
    n_train_periods = int(round(total_len*ratio_train))

    dataframeTrain = dataframeModel.iloc[:n_train_periods,:]
    dataframeTrain = dataframeTrain.fillna(0)
    dataframeTrain['const'] = 1

    if filter_data:
        for column in dataframeTrain.columns:
            dataframeTrain[column] = exponential_smoothing(dataframeTrain[column], alpha_filter)
    # Test Dataframe
    dataframeTest = dataframeModel.iloc[n_train_periods:,:]
    dataframeTest = dataframeTest.fillna(0)
    dataframeTest['const'] = 1
    if filter_data:
        for column in dataframeTest.columns:
            dataframeTest[column] = exponential_smoothing(dataframeTest[column], alpha_filter)

	dataTrain = {}
    dataTest = {}
    for item in tuple_features:
        dataTrain[item[0]] = dataframeTrain.loc[:,item[1]].values
        dataTest[item[0]] = dataframeTest.loc[:,item[1]].values
    dataTrain['index'] = dataframeTrain.index
    dataTrain['const'] = dataframeTrain.loc[:,'const'].values
    dataTest['index'] = dataframeTest.index
    dataTest['const'] = dataframeTest.loc[:,'const'].values
    
    return dataTrain, dataTest

# Fit model
def fit_model(formula_expression, dataTrain, dataTest):
    '''
        
    '''
    model = smform.OLS.from_formula(formula = formula_expression, data = dataTrain).fit()
    print(model.summary())
    
    ## Predict Results
    predictionTrain = model.predict(dataTrain)
    # predictionTest = model.predict(dataTest)
    predictionTest = model.get_prediction(dataTest)
    
    referenceTrain = dataTrain['REF']
    referenceTest = dataTest['REF']

    ## Get confidence intervals
    # For training
    # prstd, iv_l, iv_u = wls_prediction_std(model)
    st, summary_train, ss2 = summary_table(model, alpha=0.05)
    
    train_mean_se  = summary_train[:, 3]
    train_mean_ci_low, train_mean_ci_upp = summary_train[:, 4:6].T
    train_ci_low, train_ci_upp = summary_train[:, 6:8].T
    
    # For test
    summary_test = predictionTest.summary_frame(alpha=0.05)
    # test_mean, mean_se, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper
    test_mean  = summary_test.loc[:, 'mean'].values
    test_mean_ci_low = summary_test.loc[:, 'mean_ci_lower'].values
    test_mean_ci_upp = summary_test.loc[:, 'mean_ci_upper'].values
    test_ci_low = summary_test.loc[:, 'obs_ci_lower'].values
    test_ci_upp = summary_test.loc[:, 'obs_ci_upper'].values

    # Plot the stuff
    fig = plot.figure(figsize=(15,10))
    # Actual data
    plot.plot(dataTrain['index'], referenceTrain, 'k', label = 'Reference Train', alpha = 0.3)
    plot.plot(dataTest['index'], referenceTest, 'b', label = 'Reference Test', alpha = 0.3)
    # Fitted Values for Training
    plot.plot(dataTrain['index'], predictionTrain, 'g', label = 'Prediction Train')
    plot.plot(dataTrain['index'], train_ci_low, 'k--', lw=0.7, alpha = 0.5)
    plot.plot(dataTrain['index'], train_ci_upp, 'k--', lw=0.7, alpha = 0.5)
    plot.plot(dataTrain['index'], train_mean_ci_low, 'r--', lw=0.7, alpha = 0.6)
    plot.plot(dataTrain['index'], train_mean_ci_upp, 'r--', lw=0.7, alpha = 0.6)
    
    # Fitted Values for Test
    plot.plot(dataTest['index'], test_mean, 'g', label = 'Prediction Test')
    plot.plot(dataTest['index'], test_ci_low, 'k--', lw=0.7, alpha = 0.5)
    plot.plot(dataTest['index'], test_ci_upp, 'k--', lw=0.7, alpha = 0.5)
    plot.plot(dataTest['index'], test_mean_ci_low, 'r--', lw=0.7, alpha = 0.6)
    plot.plot(dataTest['index'], test_mean_ci_upp, 'r--', lw=0.7, alpha = 0.6)
    plot.title('Linear Regression Results')
    plot.ylabel('Reference/Prediction (-)')
    plot.xlabel('Date (-)')
    plot.legend(loc='best')
    plot.show()
    
    # Calculate RMSE
    rmseTrain = sqrt(mean_squared_error(predictionTrain, referenceTrain))
    rmseTest = sqrt(mean_squared_error(test_mean, referenceTest))
    
    print('------------------------')
    print('Train RMSE: %.3f' % rmseTrain)
    print('Test RMSE: %.3f' % rmseTest)

    return model, rmseTrain, rmseTest

def modelRplots(model, dataTrain, dataTest):
    
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

