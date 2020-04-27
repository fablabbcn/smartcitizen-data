# Keras LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
# Sklearn generals, SVR, RF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_squared_error
# The machine
from xgboost import XGBRegressor
# Stats Models
import statsmodels.formula.api as smform
import statsmodels.api as smapi
import statsmodels.graphics as smgraph
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.stattools import adfuller

# Plots
import matplotlib.pyplot as plot
import seaborn as sns

# Extras
from math import sqrt
import numpy as np
import pandas as pd

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
    metrics_dict['r2_score_sklearn'] = r2_score(estimation, reference) 
    # RMSD
    RMSD = sqrt((1./len(reference))*SS_Residual)
    RMSD_norm_unb = sqrt(1+np.power(sigma_norm,2)-2*sigma_norm*sqrt(rsquared))
    metrics_dict['RMSD'] = RMSD
    metrics_dict['RMSD_norm_unb'] = RMSD_norm_unb
    
    return metrics_dict

def tfuller_plot(_x, name = '', lags=None, figsize=(12, 7), lags_diff = 1):
	
	if lags_diff > 0:
		_x = _x - _x.shift(lags_diff)
		
	_x = _x.dropna()
	# _x = _x[~np.isnan(_x)]
	
	ad_fuller_result = adfuller(_x.values)
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

def prep_data_statsmodels(dataframeModel, ratio_train, fit_intercept = True):
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
	n_train_periods = int(np.round(total_len*ratio_train))

	# If fit intercept, add it
	if fit_intercept:
		dataframeModel = smapi.add_constant(dataframeModel)        
	
	dataframeTrain = dataframeModel.iloc[:n_train_periods,:]

	# Test Dataframe
	if ratio_train < 1:
		dataframeTest = dataframeModel.iloc[n_train_periods:,:]
		
		return dataframeTrain, dataframeTest, n_train_periods

	return dataframeTrain, total_len

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

def predict_statsmodels(model, data, type_model = 'OLS', plotResult = True, plotAnomalies = True, train_test = 'test'):    
	
	if 'REF' in data.columns:
		reference = data['REF']
		ref_avail = True
		mask = data.columns.str.contains('REF')
	else:
		# Do nothing
		ref_avail = False
		mask = None
		print ('No reference available')

	## Predict Results
	if train_test == 'train':
		predictionTrain = model.predict(data.loc[:,~mask])
		
		if plotResult:
			if type_model == 'OLS':
				## Get confidence intervals
				# For training
				st, summary_train, ss2 = summary_table(model, alpha=0.05)
			
				train_mean_se  = summary_train[:, 3]
				train_mean_ci_low, train_mean_ci_upp = summary_train[:, 4:6].T
				train_ci_low, train_ci_upp = summary_train[:, 6:8].T
			

				# Plot the stuff
				fig = plot.figure(figsize=(15,10))
				# Actual data
				if ref_avail:
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

		if ref_avail:
			if type_model == 'OLS': predictionTest = model.get_prediction(data.loc[:,~mask])
			elif type_model == 'RLM': predictionTest = model.predict(data.loc[:,~mask])
		else:
			if type_model == 'OLS': predictionTest = model.get_prediction(data)
			elif type_model == 'RLM': predictionTest = model.predict(data)

		
		if type_model == 'OLS':
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
				if ref_avail: plot.plot(data.index, reference, 'b', label = 'Reference Test', alpha = 0.3)
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
		else:
			test_mean = predictionTest

		if ref_avail:
			# Put test into pd dataframe 
			return pd.DataFrame(data = {'reference': reference, 'prediction': test_mean}, 
							  index = data.index)
		else:
			print ('Returning only prediction')
			return pd.DataFrame(data = {'prediction': test_mean}, 
							  index = data.index)

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
	# abs_resid = model_abs_resid.sort_values(ascending=False)
	abs_resid = np.sort(model_abs_resid)[::-1]
	abs_resid_top_3 = abs_resid[:3]
	
	# for r, i in enumerate(abs_resid_top_3):
	# # for i in abs_resid_top_3.index:
	# 	plot_lm_1.axes[0].annotate(i, 
	# 							   xy=(model_fitted_y[i], 
	# 								   model_residuals[i]));

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
	# plot_lm_4.set_figheight(height)
	# plot_lm_4.set_figwidth(width)

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

	tfuller_plot(pd.Series(model_residuals), name = 'Residuals', lags=60, lags_diff = 0)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def prep_dataframe_ML(dataframeModel, n_features, n_lags, ratio_train, verbose = True):
	# Training periods
	total_len = len(dataframeModel.index)
	n_train_periods = int(round(total_len*ratio_train))

	values = dataframeModel.values
	n_obs = n_lags * n_features
	
	## Option sensor 1 (lag 1 and no lagged prediction as feature)
	reframed = series_to_supervised(values, n_lags, 1)
	
	# drop columns we don't want
	if n_lags == 1:
		reframed = reframed.iloc[:,1:-n_features]
		n_predicted_features= 1
	else:
		# reframed_drop = reframed.iloc[:,1:]
		reframed.drop(reframed.columns[range(0,(n_features+1)*n_lags,n_features+1)], axis=1, inplace=True)
		reframed.drop(reframed.columns[range(n_obs+1, n_obs+n_features+1)], axis=1, inplace=True)
		n_predicted_features = 1
		
	values_drop = reframed.values

	# X, y
	values_drop_X = values_drop[:, :-n_predicted_features]
	values_drop_y = values_drop[:, -n_predicted_features]

	# apply scaler
	scalerX = MinMaxScaler(feature_range=(0, 1))
	scalery = MinMaxScaler(feature_range=(0, 1))
	
	values_drop_y = values_drop_y.reshape(-1, 1)
	scaledX = scalerX.fit_transform(values_drop_X)
	scaledy = scalery.fit_transform(values_drop_y)

	# train X
	train_X, test_X = scaledX[:n_train_periods], scaledX[n_train_periods:]
	train_y, test_y = scaledy[:n_train_periods], scaledy[n_train_periods:]

	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
	
	if verbose:
		print ('DataFrame has been reframed and prepared for supervised learning')
		print ('Traning X Shape {}, Training Y Shape {}, Test X Shape {}, Test Y Shape {}'.format(train_X.shape, train_y.shape, test_X.shape, test_y.shape))
	
	return train_X, train_y, test_X, test_y, scalerX, scalery

def fit_model_ML(model_type, train_X, train_y, test_X, test_y, epochs = 50, batch_size = 72, verbose = 2, plotResult = True, loss = 'mse', optimizer = 'adam', layers = ''):
	
	if model_type == 'LSTM':

		model = Sequential()
		
		if layers == '':
		
			layers = [100, 100, 100, 1]
			model.add(LSTM(layers[0], return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
			model.add(Dropout(0.2))
			model.add(LSTM(layers[1], return_sequences=True))
			model.add(LSTM(layers[2], return_sequences=False))
			model.add(Dropout(0.2))
			model.add(Dense(output_dim=layers[3]))
			model.add(Activation("linear"))
		else:

			for layer in layers:
				neurons = layer['neurons'] if 'neurons' in layer else None
				dropout_rate = layer['rate'] if 'rate' in layer else None
				activation = layer['activation'] if 'activation' in layer else None
				return_seq = layer['return_seq'] if 'return_seq' in layer else None
				input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else train_X.shape[1]
				input_dim = layer['input_dim'] if 'input_dim' in layer else train_X.shape[2]

				if layer['type'] == 'dense':
					model.add(Dense(neurons, activation=activation))
				elif layer['type'] == 'lstm':
					model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
				elif layer['type'] == 'dropout':
					model.add(Dropout(dropout_rate))

	elif model_type == 'MLP':

		# define model
		model = Sequential()

		if layers == '':
			n_nodes = 50
			n_input = 15

		else:
			n_nodes = layers[0]
			n_input = layers[1]

		model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
		model.add(Dense(1))
	   
	# Compile model 
	model.compile(loss=loss, optimizer=optimizer)

	# fit network
	history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=verbose, shuffle=False)
	if plotResult:
		# plot history
		fig = plot.figure(figsize=(10,8))
		plot.plot(history.history['loss'], label='train')
		plot.plot(history.history['val_loss'], label='test')
		plot.xlabel('Epochs (-)')
		plot.ylabel('Loss (-)')
		plot.title('Model Convergence')
		plot.legend(loc='best')
		plot.show()

	return model

def predict_ML(model, X, scalery = None):

	# Make a prediction for test
	predictions = model.predict(X)

	if scalery != None:
		inv_predictions = scalery.inverse_transform(predictions)
		inv_predictions = inv_predictions[:,-1]
	else:
		inv_predictions = predictions
	
	return inv_predictions

def get_inverse_transform_ML(y, scalery):
	
	# invert scaling for actual
	y = y.reshape((len(y), 1))
	inv_y = scalery.inverse_transform(y)
	inv_y = inv_y[:,-1]
	
	return inv_y

def prep_prediction_ML(dataframeModel, list_features, n_lags, scalerX, verbose = True):
		
	# get selected values from list    
	dataframeSupervised = dataframeModel.loc[:,list_features]
	dataframeSupervised = dataframeSupervised.dropna()
	index = dataframeSupervised.index[n_lags-1:]
	
	values = dataframeSupervised.values
	reframed = series_to_supervised(values, n_lags-1, 1)
	
	n_features = len(list_features) # There is no reference in the list
	n_obs = n_lags * n_features
		
	test = scalerX.transform(reframed.values)

	# reshape input to be 3D [samples, timesteps, features]
	test = test.reshape((test.shape[0], n_lags, n_features))
	
	if verbose:
		print ('DataFrame has been reframed and prepared for supervised learning forecasting')
		print ('Features are: {}'.format([i for i in list_features]))
		print ('Test X Shape {}'.format(test.shape))

	return test, index, n_obs

def plot_model_ML(model, dataFrameTrain = None, dataFrameTest = None, feature_list = None, model_type = None, model_name = None):
	# Plot
	fig = plot.figure(figsize=(15,10))
	
	# Actual data
	if dataFrameTrain is not None:
		plot.plot(dataFrameTrain.index, dataFrameTrain['reference'],'r', linewidth = 1, label = 'Reference Train', alpha = 0.3)
	plot.plot(dataFrameTest.index, dataFrameTest['reference'], 'b', linewidth = 1, label = 'Reference Test', alpha = 0.3)
	
	# Fitted Values for Training
	plot.plot(dataFrameTrain.index, dataFrameTrain['prediction'], 'r', linewidth = 1, label = 'Prediction Train')
	
	# Fitted Values for Test
	plot.plot(dataFrameTest.index, dataFrameTest['prediction'], 'b', linewidth = 1, label = 'Prediction Test')
	
	plot.title('{} Regression Results'.format(model_type) + model_name)
	plot.ylabel('Reference/Prediction (-)')
	plot.xlabel('Date (-)')
	plot.legend(loc='best')
	plot.show()
	
	try:
		## Model feature importances
		importances = list(model.feature_importances_)
		
		# List of tuples with variable and importance
		feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list[:], importances)]
		
		# Sort the feature importances by most important first
		feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
		
		# Print out the feature and importances 
		for pair in feature_importances:
			print ('Variable: {} Importance: {}'.format(pair[0], pair[1]))
		
		# list of x locations for plotting
		x_values = list(range(len(importances)))
		
		fig= plot.figure(figsize = (10,6))
		plot.subplot(1,2,1)
		# Make a bar chart
		plot.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
		
		# Tick labels for x axis
		plot.xticks(x_values, feature_list[:], rotation='vertical')
		
		# Axis labels and title
		plot.ylabel('Importance'); plot.xlabel('Variable'); plot.title('Variable Importances');
		
		# List of features sorted from most to least important
		sorted_importances = [importance[1] for importance in feature_importances]
		sorted_features = [importance[0] for importance in feature_importances]
		
		# Cumulative importances
		cumulative_importances = np.cumsum(sorted_importances)
		
		plot.subplot(1,2,2)
		# Make a line graph
		plot.plot(x_values, cumulative_importances, 'g-')
		
		# Draw line at 95% of importance retained
		plot.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
		
		# Format x ticks and labels
		plot.xticks(x_values, sorted_features, rotation = 'vertical')
		
		# Axis labels and title
		plot.xlabel('Variable'); plot.ylabel('Cumulative Importance'); plot.title('Cumulative Importances')

	except:
		print ('Could not plot feature importances. If model is sequential(), this is not possible')
		pass