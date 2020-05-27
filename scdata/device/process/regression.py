from scdata._config import config
from scdata.utils import std_out, dict_fmerge, clean
from pandas import DataFrame
from numpy import array

def apply_regressor(dataframe, **kwargs):
	'''
	Applies a regressor model based on a pretrained model
	Parameters
    ----------
    	model: sklearn predictor
    		Model with .predict method
		options: dict
			Options for data preprocessing. Defaults in config.model_def_opt
		variables: dict
			variables dictionary with:
				{
				'measurand': {
								'measurand-device-name': ['measurand']
								},
					'inputs': {'input-device-names': ['input-1', 'input_2', 'input-3']
					 			}
					}
    Returns
    ----------
    pandas series containing the prediction
	'''

	inputs = list()
	for device in kwargs['variables']['inputs']:
		inputs = list(set(inputs).union(set(kwargs['variables']['inputs'][device])))

	print (inputs)
	
	try:
		inputdf = dataframe[inputs].copy()
	except KeyError:
		std_out('Inputs not in dataframe', 'ERROR')
		pass
		return None

	if 'model' not in kwargs:
		std_out('Model not in inputs', 'ERROR')
	else:
		model = kwargs['model']

	if 'options' not in kwargs:
		options = config.model_def_opt
	else:
		options = dict_fmerge(config.model_def_opt, kwargs['options'])
	
	# TODO
	# Resample
	# dataframeModel = dataframeModel.resample(self.data['data_options']['frequency'], limit = 1).mean()
	
	# Remove na
	inputdf = clean(inputdf, options['clean_na'], how = 'any') 

	features = array(inputdf)
	result = DataFrame(model.predict(features)).set_index(inputdf.index)

	return result

	# feature_list = list(dataframeModel.columns)

	# if self.type == 'RF' or self.type == 'SVR':
	# 	## Get model prediction
	# 	dataframe = pd.DataFrame(self.model.predict(features_array), columns = ['prediction']).set_index(indexModel)
	# 	dataframeModel = dataframeModel.combine_first(dataframe)
	# 	data_input[prediction_name] = dataframeModel['prediction']

	# elif self.type == 'XGB':
	# 	features_array_scaled = scaler.transform(features_array)
	# 	dataframe = pd.DataFrame(self.model.predict(features_array_scaled), columns = ['prediction']).set_index(indexModel)
	# 	dataframeModel = dataframeModel.combine_first(dataframe)
	# 	data_input[prediction_name] = dataframeModel['prediction']

	# if self.plots:
	# 	# Plot
	# 	fig = plot.figure(figsize=(15,10))
	# 	# Fitted values
	# 	plot.plot(dataframeModel.index, dataframeModel['prediction'], 'b', label = 'Predicted value')
	# 	if reference is not None:
	# 		plot.plot(dataframeModel.index, dataframeModel['reference'], 'b', alpha = 0.3, label = 'Reference')
	# 	plot.grid(True)
	# 	plot.legend(loc='best')
	# 	plot.title('Model prediction for {}'.format(prediction_name))
	# 	plot.xlabel('Time (-)')
	# 	plot.ylabel(prediction_name)
	# 	plot.show()		