# Keras LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
import matplotlib.pyplot as plot
import seaborn as sns

# Sklearn generals, SVR, RF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_squared_error
import joblib

# The machine
from xgboost import XGBRegressor

import pandas as pd
import numpy as np
from numpy import concatenate
from math import sqrt

from src.data.signal_tools import metrics

import itertools
from os.path import join
import json
from src.data.recording import recording
import traceback

import statsmodels.formula.api as smform
import statsmodels.api as smapi
import statsmodels.graphics as smgraph
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.stattools import adfuller

# Others
from src.models.formulas import exponential_smoothing

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

class model_wrapper:

	def __init__(self, model_dict, verbose):
		self.name = model_dict['model_name']
		self.verbose = verbose
		self.std_out ('Beginning Model {}'.format(self.name))

		self.type = model_dict['model_type']
		self.target = model_dict['model_target']
		self.std_out ('Model type {}'.format(self.type))

		if 'hyperparameters' in model_dict.keys():
			self.hyperparameters = model_dict['hyperparameters']
			print (self.hyperparameters)
		else:
			raise SystemError ('No hyperparameters key in model input')

		if 'model_options' in model_dict.keys():
			self.options = model_dict['model_options']
		else:
			raise SystemError ('No options in model input')
		self.data = model_dict['data']

		self.model = None
		self.dataFrameTrain = None
		self.dataFrameTest = None
		self.metrics = dict()
		self.plots = model_dict['model_options']['show_plots']
		self.parameters = dict()

	def std_out(self, msg):
		if self.verbose: print (msg)

	def extract_metrics(self, train_or_test, test_name = None, test_data = None):

		# Get model metrics            
		if train_or_test == 'train':
			self.std_out ('Calculating metrics for training dataset')
			self.metrics['train'] = metrics(self.dataFrameTrain['reference'], self.dataFrameTrain['prediction'])
			self.metrics['validation'] = metrics(self.dataFrameTest['reference'], self.dataFrameTest['prediction'])
			
			# Print them out
			self.std_out ('Metrics Summary:')
			self.std_out ("{:<23} {:<7} {:<5}".format('Metric','Train','Validation'))
			for metric in self.metrics['train'].keys():
				self.std_out ("{:<20}".format(metric) +"\t" +"{:0.3f}".format(self.metrics['train'][metric]) +"\t"+ "{:0.3f}".format(self.metrics['validation'][metric]))
		
		elif train_or_test == 'test':

			self.std_out ('Calculating metrics for test dataset')
			if 'test' not in self.metrics.keys(): self.metrics['test'] = dict()

			if test_name is not None:
				if test_data is not None: 
					test_data.dropna(how = 'any', axis = 0, inplace = True)
					
					self.metrics['test'][test_name] = metrics(test_data['reference'], test_data['prediction'])
				# Print them out
				self.std_out ('Metrics Summary:')
				self.std_out ("{:<23} {:<7}".format('Metric','Test'))
				for metric in self.metrics['test'][test_name].keys():
					self.std_out ("{:<20}".format(metric) +"\t" +"{:0.3f}".format(self.metrics['test'][test_name][metric]))
		if hasattr(self.model, 'oob_score_'): self.std_out(f"oob_score: {self.model.oob_score_}")

	def training(self, model_data):

		# Get dataframe from input and reference name
		dataframeModel = model_data['data']
		reference_name = model_data['reference']
			
		labels = dataframeModel[reference_name]
		features = dataframeModel.drop(reference_name, axis = 1)
			
		# List of features for later use
		feature_list = list(features.columns)

		features = np.array(features)
		labels = np.array(labels)

		# Train model on training test
		if self.type == "RF":
			# Training and Testing Sets
			train_X, test_X, train_y, test_y = train_test_split(features, labels, 
													test_size = 1-self.hyperparameters['ratio_train'], 
													shuffle = self.hyperparameters['shuffle_split'])
		
			n_train_periods = train_X.shape[0]
			self.std_out ('Training Model {}...'.format(self.name))		
			# Create model
			if self.type == 'RF':
				self.model = RandomForestRegressor(n_estimators= self.hyperparameters['n_estimators'], 
													min_samples_leaf = self.hyperparameters['min_samples_leaf'], 
													oob_score = True, max_features = self.hyperparameters['max_features'])
			elif self.type == 'SVR':
				self.model = SVR(kernel='rbf')



			# Fit model
			# print (np.isnan(np.sum(train_X)))
			# print (np.isnan(np.sum(train_y)))
			self.model.fit(train_X, train_y)
			
			## Get model prediction for training dataset
			predictionTrain = predict_ML(self.model, features[:n_train_periods])
			
			self.dataFrameTrain = pd.DataFrame(data = {'reference': labels[:n_train_periods], 'prediction': predictionTrain}, 
												index = dataframeModel.index[:n_train_periods])

			## Get model prediction for training dataset			
			predictionTest = predict_ML(self.model, features[n_train_periods:])

			self.dataFrameTest = pd.DataFrame(data = {'reference': labels[n_train_periods:], 'prediction': predictionTest}, 
												index = dataframeModel.index[n_train_periods:])

			if self.plots:
				plot_model_ML(self.model, self.dataFrameTrain, self.dataFrameTest, feature_list, self.type, self.name)
		
		elif self.type == "XGB":
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features)
			
			train_X, test_X, train_y, test_y = train_test_split(features_scaled, labels, 
													test_size = 1-self.hyperparameters['ratio_train'], 
													shuffle = False)
			# Fit XGB regressor
			self.model = XGBRegressor(n_threads = -1, param = self.hyperparameters['param'])

			self.model.fit(train_X, train_y)
			predictionTrain = self.model.predict(train_X)
			predictionTest = self.model.predict(test_X)
			n_train_periods = train_X.shape[0]

			# Save scaler
			self.parameters['scaler'] = scaler

			self.dataFrameTrain = pd.DataFrame(data = {'reference': labels[:n_train_periods], 'prediction': predictionTrain}, 
												index = dataframeModel.index[:n_train_periods])

			self.dataFrameTest = pd.DataFrame(data = {'reference': labels[n_train_periods:], 'prediction': predictionTest}, 
												index = dataframeModel.index[n_train_periods:])

			if self.plots:
				plot_model_ML(self.model, self.dataFrameTrain, self.dataFrameTest, feature_list, self.type, self.name)

		elif self.type == "OLS" or self.type == 'GLM' or self.type == 'RLM':
			
			# OLS can be done with a formula, and the model needs to be renamed accordingly
			for item in self.data['features'].keys():
				target_name = item
				for column in dataframeModel.columns:
					if self.data['features'][item] in column: 
						current_name = column
						dataframeModel.rename(columns={current_name: target_name}, inplace=True)
						break
			
			dataTrain, dataTest, n_train_periods = prep_data_statsmodels(dataframeModel, self.hyperparameters['ratio_train'])

			## Model Fit
			if 'expression' in self.data.keys(): expression = self.data['expression']
			else: expression = None
			if 'print_summary' in self.options: print_summary = self.options['print_summary']
			else: print_summary = False
			
			if self.type == "OLS":
				# The constant is only added if the 
				if expression is not None:
					self.model = smform.ols(formula = expression, data = dataTrain).fit()
				else:
					mask = dataTrain.columns.str.contains('REF')

					self.model = smapi.OLS(dataTrain.loc[:,mask].values, dataTrain.loc[:,~mask].values).fit()
			elif self.type == "RLM":
				mask = dataTrain.columns.str.contains('REF')
				self.model = smapi.RLM(dataTrain.loc[:,mask].values, dataTrain.loc[:,~mask].values, M=smapi.robust.norms.HuberT()).fit()
			if print_summary:
				print(self.model.summary())

			## Get model prediction
			self.dataFrameTrain = predict_statsmodels(self.model, dataTrain, type_model = self.type, plotResult = self.plots, plotAnomalies = False, train_test = 'train')

			self.dataFrameTest = predict_statsmodels(self.model, dataTest, type_model = self.type, plotResult = self.plots, plotAnomalies = False, train_test = 'test')

			if self.plots and self.type == "OLS":
				# plot_OLS_coeffs(self.model)
				model_R_plots(self.model, dataTrain, dataTest)

		# Not supported for now
		# elif self.type == "LSTM":
		# 	n_features = len(feature_list)
			
		# 	# Data Split
		# 	train_X, train_y, test_X, test_y, scalerX, scalery = prep_dataframe_ML(dataframeModel, n_features, self.hyperparameters['n_lags'], self.hyperparameters['ratio_train'])
			
		# 	index = dataframeModel.index
		# 	n_train_periods = train_X.shape[0]
			
		# 	# Model Fit
		# 	self.std_out ('Model training...')

		# 	# Check if we have specific layers
		# 	if 'layers' in self.hyperparameters.keys():
		# 		layers = self.hyperparameters['layers']
		# 	else:
		# 		layers = ''

		# 	self.model = fit_model_ML('LSTM', train_X, train_y, 
		# 						   test_X, test_y, 
		# 						   epochs = self.hyperparameters['epochs'], batch_size = self.hyperparameters['batch_size'], 
		# 						   verbose = self.hyperparameters['verbose'], plotResult = self.plots, 
		# 						   loss = self.hyperparameters['loss'], optimizer = self.hyperparameters['optimizer'], layers = layers)
			
		# 	inv_train_y = get_inverse_transform_ML(train_y, scalery)
		# 	inv_test_y = get_inverse_transform_ML(test_y, scalery)

		# 	predictionTrain = predict_ML(self.model, train_X, scalery)
		# 	predictionTest = predict_ML(self.model, test_X, scalery)

		# 	self.dataFrameTrain = pd.DataFrame(data = {'reference': inv_train_y, 'prediction': predictionTrain}, 
		# 										index = dataframeModel.index[:n_train_periods])

		# 	self.dataFrameTest= pd.DataFrame(data = {'reference': inv_test_y, 'prediction': predictionTest}, 
		# 										index = index[n_train_periods+self.hyperparameters['n_lags']:])
						
		# 	self.parameters['scalerX'] = scalerX
		# 	self.parameters['scalery'] = scalery

		# 	if self.plots:
		# 		plot_model_ML(self.model, self.dataFrameTrain, self.dataFrameTest, feature_list, self.type, self.name)

		self.parameters['n_train_periods'] = n_train_periods

		if self.options['extract_metrics']: self.extract_metrics('train')
		else: self.metrics = None

	def predict_channels(self, data_input, prediction_name, reference = None, reading_name = None):
		
		# Get specifics
		if self.type == 'LSTM':
			scalerX_predict = self.parameters['scalerX']
			scalery_predict = self.parameters['scalery']
			n_lags = self.hyperparameters['n_lags']
			self.std_out('Loading parameters for ')
		elif self.type == 'RF' or self.type == 'SVR' or self.type == 'OLS':
			self.std_out('No specifics for {} type'.format(self.type))
		elif self.type == 'XGB':
			scaler = self.parameters['scaler']


		list_features = list()
		for item in self.data['features']:
			if item != 'REF':
				list_features.append(self.data['features'][item])
				if self.data['features'][item] not in data_input.columns:
					self.std_out('{} not in input data. Cannot predict using this model'.format(self.data['features'][item]))
					break

		self.std_out('Preparing devices from prediction')
		dataframeModel = data_input.loc[:, list_features]
		dataframeModel = dataframeModel.apply(pd.to_numeric,errors='coerce')   
		
		# Resample
		dataframeModel = dataframeModel.resample(self.options['target_raster'], limit = 1).mean()
		
		# Remove na
		if self.options['clean_na']:

			if self.options['clean_na_method'] == 'fill':
				dataframeModel = dataframeModel.fillna(method='bfill').fillna(method='ffill')
			elif self.options['clean_na_method'] == 'drop':
				dataframeModel.dropna(axis = 0, how = 'any', inplace = True)

		indexModel = dataframeModel.index

		# List of features for later use
		feature_list = list(dataframeModel.columns)
		features_array = np.array(dataframeModel)

		if self.type == 'RF' or self.type == 'SVR':
			## Get model prediction
			dataframe = pd.DataFrame(self.model.predict(features_array), columns = ['prediction']).set_index(indexModel)
			dataframeModel = dataframeModel.combine_first(dataframe)
			data_input[prediction_name] = dataframeModel['prediction']

		elif self.type == 'XGB':
			features_array_scaled = scaler.transform(features_array)
			dataframe = pd.DataFrame(self.model.predict(features_array_scaled), columns = ['prediction']).set_index(indexModel)
			dataframeModel = dataframeModel.combine_first(dataframe)
			data_input[prediction_name] = dataframeModel['prediction']
		
		# Not supported for now
		# elif self.type == 'LSTM':
		# 	# To fix
		# 	test_X, index_pred, n_obs = prep_prediction_ML(dataframeModel, list_features, n_lags, scalerX_predict, verbose = self.verbose)
		# 	prediction = predict_ML(self.model, test_X, scalery_predict)

		# 	dataframe = pd.DataFrame(prediction, columns = [prediction_name]).set_index(index_pred)
		# 	data_input[prediction_name] = dataframe.loc[:,prediction_name]
		
		elif self.type == 'OLS' or self.type == 'RLM':

			if 'expression' in self.data.keys():

				# Rename to formula
				for item in features.keys():
					dataframeModel.rename(columns={features[item]: item}, inplace=True)

			## Predict the model results
			datapredict, _ = prep_data_statsmodels(dataframeModel, 1)
			prediction = predict_statsmodels(self.model, datapredict, type_model = self.type, plotResult = self.plots, plotAnomalies = False, train_test = 'test')
			
			dataframe = pd.DataFrame(prediction, columns = ['prediction']).set_index(indexModel)
			dataframeModel = dataframeModel.combine_first(dataframe)
			print(datapredict.columns)
			
			data_input[prediction_name] = prediction
		
		self.std_out('Channel {} prediction finished'.format(prediction_name))
		
		if reference is not None:	
			
			min_date_combination = max(reference.index[0], data_input.index[0])		
			max_date_combination = min(reference.index[-1], data_input.index[-1])

			dataframeModel = dataframeModel.combine_first(pd.DataFrame(reference.values, columns=['reference']).set_index(reference.index))

			dataframeModel = dataframeModel[dataframeModel.index>min_date_combination]
			dataframeModel = dataframeModel[dataframeModel.index<max_date_combination]
			dataframeModel.dropna(axis = 0, how = 'any', inplace = True)

			if self.options['extract_metrics']: self.extract_metrics('test', reading_name, dataframeModel) 

		if self.plots:
			# Plot
			fig = plot.figure(figsize=(15,10))
			# Fitted values
			plot.plot(dataframeModel.index, dataframeModel['prediction'], 'b', label = 'Predicted value')
			if reference is not None:
				plot.plot(dataframeModel.index, dataframeModel['reference'], 'b', alpha = 0.3, label = 'Reference')
			plot.grid(True)
			plot.legend(loc='best')
			plot.title('Model prediction for {}'.format(prediction_name))
			plot.xlabel('Time (-)')
			plot.ylabel(prediction_name)
			plot.show()

		return data_input

	def export(self, directory):

		modelDir = join(directory, self.target)
		summaryDir = join(directory, 'summary.json')
		filename = join(modelDir, self.name)

		self.std_out('Saving metrics')
		joblib.dump(self.metrics, filename + '_metrics.sav')
		self.std_out('Saving hyperparameters')
		joblib.dump(self.hyperparameters, filename + '_hyperparameters.sav')
		self.std_out('Saving features')
		joblib.dump(self.data, filename + '_features.sav')
		
		if self.options['export_model_file']:
			self.std_out('Dumping model file')
			if self.type == 'LSTM':
				model_json = self.model.to_json()
				with open(filename + "_model.json", "w") as json_file:
					json_file.write(model_json)
				self.std_out('Dumping model weights')
				self.model.save_weights(filename + "_model.h5")
				
			elif self.type == 'RF' or self.type == 'SVR' or self.type == 'OLS':
				joblib.dump(self.model, filename + '_model.sav', compress = 3)
		
			self.std_out("Model:" + self.name + "\nSaved in " + modelDir)
		
		summary = json.load(open(summaryDir, 'r'))
		if self.target not in summary.keys(): summary[self.target] = dict()
		summary[self.target][self.name] = dict()
		summary[self.target][self.name] = self.type

		with open(summaryDir, 'w') as json_file:
			json_file.write(json.dumps(summary))
			json_file.close()

		self.std_out("Model included in summary")