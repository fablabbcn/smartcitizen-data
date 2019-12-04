from src.models.formulas import *	
from src.saf import *
from src.models.model_tools import *

class model_wrapper ():

	def __init__(self, model_dict, verbose):
		saf.__init__(self, verbose)
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
		dataframeModel = dataframeModel.resample(self.options['frequency'], limit = 1).mean()
		
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