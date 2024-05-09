from scdata._config import config
from scdata.tools.custom_logger import logger
from scdata.tools.dictmerge import dict_fmerge
from scdata.tools.cleaning import clean
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
		logger.error('Inputs not in dataframe')
		pass
		return None

	if 'model' not in kwargs:
		logger.error('Model not in inputs')
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