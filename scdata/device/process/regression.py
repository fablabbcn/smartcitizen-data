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
			Options for data preprocessing. Defaults in config._model_def_opt
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

	try:
		inputdf = dataframe[inputs].copy()
		inputdf = inputdf.reindex(sorted(inputdf.columns), axis=1)
	except KeyError:
		std_out('Inputs not in dataframe', 'ERROR')
		pass
		return None

	if 'model' not in kwargs:
		std_out('Model not in inputs', 'ERROR')
	else:
		model = kwargs['model']

	if 'options' not in kwargs:
		options = config._model_def_opt
	else:
		options = dict_fmerge(config._model_def_opt, kwargs['options'])
	
	# Remove na
	inputdf = clean(inputdf, options['clean_na'], how = 'any') 

	features = array(inputdf)
	result = DataFrame(model.predict(features)).set_index(inputdf.index)

	return result