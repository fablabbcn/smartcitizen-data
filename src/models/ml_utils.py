import pandas as pd

# Combine all data in one dataframe
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder() 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from src.data.test_utils import combine_data
import ipywidgets as widgets

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
import matplotlib.pyplot as plot

from sklearn.metrics import r2_score, median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error #, mean_squared_log_error 

from numpy import concatenate
from math import sqrt

from src.models.formula_utils import exponential_smoothing
import numpy as np

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

def predict_ML(model, X, y, index, scalery = None):

	# Make a prediction for test
	predictions = model.predict(X)

	if scalery != None:
		inv_predictions = scalery.inverse_transform(predictions)
		inv_predictions = inv_predictions[:,-1]
	else:
		inv_predictions = predictions

	errors = abs(inv_predictions - y)
	fig = plot.figure(figsize=(10,8))
	plot.plot(y, label = 'Reference')
	plot.plot(inv_predictions, label = 'Prediction')
	plot.plot(errors, label = 'Errors')
	rerror = np.maximum(np.minimum(np.divide(errors, y),1),-1)
	plot.plot(rerror, label = 'Relative Error')
	plot.legend(loc='best')
	mape = 100 * np.mean(rerror)
	accuracy = 100 - mape
	print('Model Performance')
	print('\tAverage Error: {:0.4f}.'.format(np.mean(errors)))
	print('\tAccuracy = {:0.2f}%.'.format(accuracy))

	dataFrame = pd.DataFrame(data = {'reference': y, 'prediction': inv_predictions}, 
							  index = index)
	
	return dataFrame

def get_inverse_transform_ML(y, scalery):
	
	# invert scaling for actual
	y = y.reshape((len(y), 1))
	inv_y = scalery.inverse_transform(y)
	inv_y = inv_y[:,-1]
	
	return inv_y

def prep_prediction_ML(dataframeModel, list_features, n_lags, alpha_filter, scalerX, verbose = True):
		
	# get selected values from list    
	dataframeSupervised = dataframeModel.loc[:,list_features]
	dataframeSupervised = dataframeSupervised.dropna()
	index = dataframeSupervised.index[n_lags-1:]

	if alpha_filter<1:
		for column in dataframeSupervised.columns:
			dataframeSupervised[column] = exponential_smoothing(dataframeSupervised[column], alpha_filter)
	
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
		
		fig= plot.figure(figsize = (12,6))
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