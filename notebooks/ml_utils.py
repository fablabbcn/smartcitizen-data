import pandas as pd

# Combine all data in one dataframe
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder() 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from test_utils import combine_data
import ipywidgets as widgets

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
import matplotlib.pyplot as plot

from sklearn.metrics import r2_score, median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error#, mean_squared_log_error 

from numpy import concatenate
from math import sqrt

from formula_utils import exponential_smoothing

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

def prep_dataframe_ML(dataframeModel, min_date, max_date, list_features, n_lags, ratio_train, alpha_filter, reference_name, verbose = True):

    ## Trim dates
    dataframeModel = dataframeModel[dataframeModel.index > min_date]
    dataframeModel = dataframeModel[dataframeModel.index < max_date]
        
    # get selected values from list
    dataframeSupervised = dataframeModel.loc[:,list_features]
    dataframeSupervised = dataframeSupervised.dropna()

    # Training periods
    total_len = len(dataframeSupervised.index)
    n_train_periods = int(round(total_len*ratio_train))

    if alpha_filter<1:
        for column in dataframeSupervised.columns:
            dataframeSupervised[column] = exponential_smoothing(dataframeSupervised[column], alpha_filter)
    
    index = dataframeSupervised.index
    values = dataframeSupervised.values

    n_features = len(list_features) - 1
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
    scaledX = scalerX.fit_transform(values_drop_X)
    scaledy = scalery.fit_transform(values_drop_y)

    # train X
    train_X, test_X = scaledX[:n_train_periods], scaledX[n_train_periods:]
    train_y, test_y = scaledy[:n_train_periods], scaledy[n_train_periods:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
    
    if verbose:
        print 'DataFrame has been reframed and prepared for supervised learning'
        print 'Reference is: {}'.format(reference_name)
        print 'Features are: {}'.format([i for i in list_features[1:]])
        print 'Traning X Shape {}, Training Y Shape {}, Test X Shape {}, Test Y Shape {}'.format(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    return index, train_X, train_y, test_X, test_y, scalerX, scalery, n_train_periods

def fit_model_ML(train_X, train_y, test_X, test_y, epochs = 50, batch_size = 72, verbose = 2, plotResult = True, loss = 'mse', optimizer = 'adam', layers = ''):
    
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
			if layer['type'] == 'lstm':
				model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				model.add(Dropout(dropout_rate))

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


def predict_ML(model, test_X, n_lags, scalery):

    # Make a prediction for test
    yhat = model.predict(test_X)
    inv_yhat = scalery.inverse_transform(yhat)
    inv_yhat = inv_yhat[:,-1]

    return inv_yhat

def get_inverse_transform_ML(test_y, n_lags, scalery):
    
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = scalery.inverse_transform(test_y)
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
        print 'DataFrame has been reframed and prepared for supervised learning forecasting'
        print 'Features are: {}'.format([i for i in list_features])
        print 'Test X Shape {}'.format(test.shape)
    
    return test, index, n_obs