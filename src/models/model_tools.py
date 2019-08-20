# Keras LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
import matplotlib.pyplot as plot

# Sklearn generals, SVR, RF
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error #, mean_squared_log_error 
import joblib

from src.models.linear_regression_tools import prep_data_OLS, fit_model_OLS, predict_OLS
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

def plot_model_ML(model, dataFrameTrain, dataFrameTest, feature_list, model_type, model_name):
	# Plot
	fig = plot.figure(figsize=(15,10))
	
	# Actual data
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
		
		fig= plot.figure(figsize = (15,8))
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
		self.std_out ('Model type {}'.format(self.type))

		if 'hyperparameters' in model_dict.keys():
			self.hyperparameters = model_dict['hyperparameters']
		else:
			raise SystemError ('No hyperparameters key in model input')

		if 'options' in model_dict.keys():
			self.options = model_dict['options']
		else:
			raise SystemError ('No options in model input')
		self.data = model_dict['data']

		self.model = None
		self.dataFrameTrain = None
		self.dataFrameTest = None
		self.metrics = None
		self.plots = model_dict['options']['show_plots']
		self.parameters = dict()

	def std_out(self, msg):
		if self.verbose: print (msg)

	def extract_metrics(self):             
		# Get model metrics            
		self.std_out ('Calculating Metrics...')
		self.metrics = dict()
		self.metrics['train'] = metrics(self.dataFrameTrain['reference'], self.dataFrameTrain['prediction'])
		self.metrics['test'] = metrics(self.dataFrameTest['reference'], self.dataFrameTest['prediction'])
		
		# Print them out
		self.std_out ('Metrics Summary:')
		self.std_out ("{:<23} {:<7} {:<5}".format('Metric','Train','Test'))
		for metric in self.metrics['train'].keys():
			self.std_out ("{:<20}".format(metric) +"\t" +"{:0.3f}".format(self.metrics['train'][metric]) +"\t"+ "{:0.3f}".format(self.metrics['test'][metric]))
		
		return self.metrics

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
		if self.type == "RF" or self.type == 'SVR':
				
			# Training and Testing Sets
			train_X, test_X, train_y, test_y = train_test_split(features, labels, random_state = 42, 
													test_size = 1-self.hyperparameters['ratio_train'], shuffle = self.hyperparameters['shuffle_split'])
		
			n_train_periods = train_X.shape[0]
			
			# Create model
			if self.type == 'RF':
				self.model = RandomForestRegressor(n_estimators= self.hyperparameters['n_estimators'], random_state = 42)
			elif self.type == 'SVR':
				self.model = SVR(kernel='rbf')
			
			self.std_out ('Training Model {}...'.format(self.name))
			# Fit model
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

		elif self.type == "OLS":
			# OLS can be done with a formula, and the model needs to be renamed accordingly
			reference_device = self.data['reference_device']
			train_dataset = list(self.data['train'].keys())[0]
			device = self.data['train'][train_dataset]
			
			for item in self.data['features'].keys():
				target_name = item
				if item == 'REF':
					current_name = '_'.join([self.data['features'][item], reference_device])
				else:
					current_name = '_'.join([self.data['features'][item], device])

				dataframeModel.rename(columns={current_name: target_name}, inplace=True)
			
			dataTrain, dataTest, n_train_periods = prep_data_OLS(dataframeModel, self.hyperparameters['ratio_train'])
			
			## Model Fit
			expression = None
			if 'expression' in self.data.keys(): expression = self.data['expression']
			
			self.model = fit_model_OLS(expression, dataTrain, printSummary = False)

			## Get model prediction
			self.dataFrameTrain = predict_OLS(self.model, 
											  dataTrain, 
											  plotResult = self.plots, 
											  plotAnomalies = False, 
											  train_test = 'train')

			self.dataFrameTest = predict_OLS(self.model, 
											  dataTest, 
											  plotResult = self.plots, 
											  plotAnomalies = False, 
											  train_test = 'test')

			if self.plots:
				plot_OLS_coeffs(self.model)
				model_R_plots(self.model, dataTrain, dataTest)

		elif self.type == "LSTM":
			n_features = len(feature_list)
			
			# Data Split
			train_X, train_y, test_X, test_y, scalerX, scalery = prep_dataframe_ML(dataframeModel, 
																				   n_features, 
																				   self.hyperparameters['n_lags'], 
																				   self.hyperparameters['ratio_train'])
			
			index = dataframeModel.index
			n_train_periods = train_X.shape[0]
			
			# Model Fit
			self.std_out ('Model training...')

			# Check if we have specific layers
			if 'layers' in self.hyperparameters.keys():
				layers = self.hyperparameters['layers']
			else:
				layers = ''

			self.model = fit_model_ML('LSTM', train_X, train_y, 
								   test_X, test_y, 
								   epochs = self.hyperparameters['epochs'], batch_size = self.hyperparameters['batch_size'], 
								   verbose = self.hyperparameters['verbose'], plotResult = self.plots, 
								   loss = self.hyperparameters['loss'], optimizer = self.hyperparameters['optimizer'], layers = layers)
			
			inv_train_y = get_inverse_transform_ML(train_y, scalery)
			inv_test_y = get_inverse_transform_ML(test_y, scalery)

			predictionTrain = predict_ML(self.model, train_X, scalery)
			predictionTest = predict_ML(self.model, test_X, scalery)

			self.dataFrameTrain = pd.DataFrame(data = {'reference': inv_train_y, 'prediction': predictionTrain}, 
												index = dataframeModel.index[:n_train_periods])

			self.dataFrameTest= pd.DataFrame(data = {'reference': inv_test_y, 'prediction': predictionTest}, 
												index = index[n_train_periods+self.hyperparameters['n_lags']:])
						
			self.parameters['scalerX'] = scalerX
			self.parameters['scalery'] = scalery

			if self.plots:
				plot_model_ML(self.model, self.dataFrameTrain, self.dataFrameTest, feature_list, self.type, self.name)

		self.parameters['n_train_periods'] = n_train_periods

		if self.options['extract_metrics']: self.metrics = self.extract_metrics()
		else: self.metrics = None

	def predict_channels(self, data_input, prediction_name):
		
		# Get specifics
		if self.type == 'LSTM':
			scalerX_predict = self.parameters['scalerX']
			scalery_predict = self.parameters['scalery']
			n_lags = self.hyperparameters['n_lags']
			self.std_out('Loading parameters for ')
		elif self.type == 'RF' or self.type == 'SVR' or self.type == 'OLS':
			self.std_out('No specifics for {} type'.format(self.type))

		list_features = list()
		for item in self.data['features']:
			if item != 'REF':
				list_features.append(self.data['features'][item])
				if self.data['features'][item] not in data_input.columns:
					self.std_out('{} not in input data. Cannot predict using this model'.format(self.data['features'][item]))
					break

		self.std_out('Preparing devices from prediction')
		dataframeModel = data_input.loc[:, list_features]
		# dataframeModel = dataframeModel.apply(pd.to_numeric,errors='coerce')   
		
		# # Resample
		# dataframeModel = dataframeModel.resample(self.options['target_raster']).mean()
		
		# # Remove na
		# if self.options['clean_na']:
		# 	if self.options['clean_na_method'] == 'fill':
		# 		dataframeModel = dataframeModel.fillna(method='bfill').fillna(method='ffill')
		# 	elif self.options['clean_na_method'] == 'drop':
		# 		dataframeModel = dataframeModel.dropna()

		indexModel = dataframeModel.index

		# List of features for later use
		feature_list = list(dataframeModel.columns)
		features_array = np.array(dataframeModel)

		if self.type == 'RF' or self.type == 'SVR':
			## Get model prediction
			dataframe = pd.DataFrame(self.model.predict(features_array), columns = ['prediction']).set_index(indexModel)
			dataframeModel = dataframeModel.combine_first(dataframe)
			data_input[prediction_name] = dataframeModel['prediction']

		elif self.type == 'LSTM':
			# To fix
			test_X, index_pred, n_obs = prep_prediction_ML(dataframeModel, list_features, n_lags, scalerX_predict, verbose = self.verbose)
			prediction = predict_ML(self.model, test_X, scalery_predict)

			dataframe = pd.DataFrame(prediction, columns = [prediction_name]).set_index(index_pred)
			data_input[prediction_name] = dataframe.loc[:,prediction_name]
		
		elif self.type == 'OLS':

			if 'formula' in self.options.keys(): 

				# Rename to formula
				for item in features.keys():
					dataframeModel.rename(columns={features[item]: item}, inplace=True)

			## Predict the model results
			datapredict, _ = prep_data_OLS(dataframeModel, features, 1)
			prediction = predict_OLS(model, datapredict, False, False, 'test')

			data_input[prediction_name] = prediction
		
		self.std_out('Channel {} prediction finished'.format(prediction_name))

		if self.plots:
			# Plot
			fig = plot.figure(figsize=(15,10))
			# Fitted values
			plot.plot(data_input.index, data_input[prediction_name], 'b', label = 'Predicted value')
			plot.grid(True)
			plot.legend(loc='best')
			plot.title('Model prediction for {}'.format(prediction_name))
			plot.xlabel('Time (-)')
			plot.ylabel(prediction_name)
			plot.show()

		return data_input

	def export(self, directory):

		modelDir = join(directory, self.options['model_target'])
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
		summary[self.options['model_target']][self.name] = dict()
		summary[self.options['model_target']][self.name] = self.type

		with open(summaryDir, 'w') as json_file:
			json_file.write(json.dumps(summary))
			json_file.close()

		self.std_out("Model included in summary")