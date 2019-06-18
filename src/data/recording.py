from src.data.test_utils import *
from src.data.api_utils import *
from os.path import join
from sklearn.externals import joblib
import matplotlib.pyplot as plot
import json

from sklearn.ensemble import RandomForestRegressor
from src.models.ml_utils import prep_prediction_ML, predict_ML
from src.models.linear_regression_utils import predict_OLS, prep_data_OLS

import pandas as pd
import numpy as np

## Initialise paths and working directories
from os import pardir, getcwd
from os.path import join, abspath, normpath, basename
from src.data.test_utils import getSensorNames, getTests
from src.models.pollutant_cal_utils import *
from src.models.formula_utils import *

class recording:

	def __init__(self):
		
		self.readings = dict()
		self.rootDirectory = abspath(abspath(join(getcwd(), pardir)))
		self.dataDirectory = join(self.rootDirectory, 'data')
		self.interimDirectory = join(self.dataDirectory, 'interim')
		self.modelDirectory = join(self.rootDirectory, 'models')
		currentSensorsh = ('https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/dev/lib/Sensors/Sensors.h')
		self.availablereadings = getTests(self.dataDirectory)
		
		try:
			self.currentSensorNames = getSensorNames(currentSensorsh, join(self.dataDirectory, 'interim'))
		except:
			raise SystemError('Sensor names could not be loaded')
		else:
			print ('Sensor names loaded OK')

	def reload_tests(self):
		self.availablereadings = getTests(self.dataDirectory)
	
	def load_recording_CSV(self, reading_name, source_id, target_raster = '1Min', clean_na = True, clean_na_method = 'fill', load_processed = True):
		data = loadTest(source_id, target_raster, self.currentSensorNames, clean_na, clean_na_method, self.dataDirectory, load_processed)
		self.readings[reading_name] = dict()
		self.readings[reading_name] = data[reading_name]

		# Set flag
		self.readings[reading_name]['ready_to_model'] = False

	def load_recording_API(self, reading_name, source_id, min_date, max_date, target_raster = '1Min', dataDirectory = '', clean_na = True, clean_na_method = 'fill'):
		data = getReadingsAPI(source_id, target_raster, min_date, max_date, self.currentSensorNames, self.dataDirectory, clean_na, clean_na_method)
		
		# Case for non merged API to CSV
		if reading_name not in self.readings.keys():
			self.readings[reading_name] = dict()
			self.readings[reading_name] = data
		
		# Case for merged API to CSV
		else:
			for key in data['devices'].keys():
				self.readings[reading_name]['devices'][key] = data['devices'][key] 

		# Set flag
		self.readings[reading_name]['ready_to_model'] = False

	def del_recording(self, reading_name):
		if reading_name in self.readings.keys():
			self.readings.pop(reading_name)
		print ('Deleting', reading_name)

	def clear_recordings(self):
		self.readings.clear()
		print ('Clearing recordings')

	def combine_readings(self, reading_name, name_combined_data = 'COMBINED_DEVICES'):
		try: 
			## Since we don't know if there are more or less channels than last time
			## (and tbh, I don't feel like checking), remove the key
			self.readings[reading_name]['devices'].pop(name_combined_data, None)
			ignore_keys = []
			if 'models' in self.readings[reading_name].keys():
				ignore_keys = self.readings[reading_name]['models'].keys()

			print ('\tIgnoring keys', ignore_keys)
			## And then add it again
			dataframe = combine_data(self.readings[reading_name]['devices'], True, ignore_keys)

			self.readings[reading_name]['devices'][name_combined_data] = dict()
			self.readings[reading_name]['devices'][name_combined_data]['data'] = dict()
			self.readings[reading_name]['devices'][name_combined_data]['data'] = dataframe

		except:
			print('\tError ocurred. Review data')
			return False
		else:
			print('\n\tData combined successfully')
			return True

	def prepare_dataframe_model(self, tuple_features, reading_name, min_date, max_date, model_name, name_combined_data = 'COMBINED_DEVICES', clean_na = True, clean_na_method = 'fill' , target_raster = '1Min'):
	
		if self.combine_readings(reading_name):

			## Send only the needed features
			list_features = list()
			try:
				# Create the list of features needed
				for item in tuple_features:
					item_name = item[1] + '_' + item[2]
					list_features.append(item_name)
				dataframeModel = self.readings[reading_name]['devices'][name_combined_data]['data'].loc[:,list_features]
				
				dataframeModel = dataframeModel.apply(pd.to_numeric,errors='coerce')   
				# Resample
				dataframeModel = dataframeModel.resample(target_raster).mean()
				
				# Remove na
				if clean_na:
					if clean_na_method == 'fill':
						dataframeModel = dataframeModel.fillna(method='bfill').fillna(method='ffill')
					elif clean_na_method == 'drop':
						dataframeModel = dataframeModel.dropna()
				
				if min_date != None:
					dataframeModel = dataframeModel[dataframeModel.index > min_date]
				if max_date != None:
					dataframeModel = dataframeModel[dataframeModel.index < max_date]

				if model_name != None:
					# Don't create the model structure, since we are predicting
					if 'models' not in self.readings[reading_name].keys():
						print ('\tCreating models dict')
						self.readings[reading_name]['models']=dict()

					self.readings[reading_name]['models'][model_name]=dict()
					self.readings[reading_name]['models'][model_name]['data'] = dataframeModel
					self.readings[reading_name]['models'][model_name]['features'] = tuple_features

					for item in tuple_features:
						if item[0] == 'REF':
							reference = dataframeModel.loc[:,item[1] + '_' + item[2]]
							reference_name = reference.name
							self.readings[reading_name]['models'][model_name]['reference'] = reference_name
							print ('\tModel reference is', reference_name)

					# Set flag
					self.readings[reading_name]['ready_to_model'] = True
				
			except:
				if item_name not in self.readings[reading_name]['devices'][name_combined_data]['data'].columns:
					print ('{} not in {}'.format(item_name, reading_name))
			else: 
				print ('\tDataframe model generated successfully')

	def archive_model(self, reading_name, model_name, metrics_model, dataframe, model, model_type, model_target, ratio_train, formula = '', n_lags = None, scalerX = None, scalery = None, shuffle_split = None, alpha_filter = None):
		try:
			# Metrics
			self.readings[reading_name]['models'][model_name]['metrics'] = metrics_model
	
			# Model and modelType
			self.readings[reading_name]['models'][model_name]['model'] = model
			self.readings[reading_name]['models'][model_name]['model_type'] = model_type
			self.readings[reading_name]['models'][model_name]['model_target'] = model_target
			self.readings[reading_name]['models'][model_name]['formula'] = formula
	
			# Parameters
			self.readings[reading_name]['models'][model_name]['parameters'] = dict()
			self.readings[reading_name]['models'][model_name]['parameters']['ratio_train'] = ratio_train
			if scalerX != None:
				self.readings[reading_name]['models'][model_name]['parameters']['scalerX'] = scalerX
			if scalery != None:
				self.readings[reading_name]['models'][model_name]['parameters']['scalery'] = scalery
			if n_lags != None:
				self.readings[reading_name]['models'][model_name]['parameters']['n_lags'] = n_lags
			if shuffle_split != None:
				self.readings[reading_name]['models'][model_name]['parameters']['shuffle_split'] = shuffle_split
			if alpha_filter != None:
				self.readings[reading_name]['models'][model_name]['parameters']['alpha_filter'] = alpha_filter
			
			# Dataframe
			self.readings[reading_name]['devices'][model_name] = dict()
			self.readings[reading_name]['devices'][model_name]['data'] = dataframe
			
		
		except:
			print ('Problem occured')
			pass
		else:
			print('Model archived correctly')

	def export_model(self, reading_name, model_name):
		model_target = self.readings[reading_name]['models'][model_name]['model_target']
		model_type = self.readings[reading_name]['models'][model_name]['model_type']
		model = self.readings[reading_name]['models'][model_name]['model']

		modelDir = join(self.modelDirectory, model_target)
		summaryDir = join(self.modelDirectory, 'summary.json')
		filename = join(modelDir, model_name)

		joblib.dump(self.readings[reading_name]['models'][model_name]['metrics'], filename + '_metrics.sav')
		joblib.dump(self.readings[reading_name]['models'][model_name]['parameters'], filename + '_parameters.sav')
		joblib.dump( self.readings[reading_name]['models'][model_name]['features'], filename + '_features.sav')

		if model_type == 'LSTM':
			model_json = model.to_json()
			with open(filename + "_model.json", "w") as json_file:
				json_file.write(model_json)
			model.save_weights(filename + "_model.h5")
			
		elif model_type == 'RF' or model_type == 'SVR' or model_type == 'OLS':
			joblib.dump(model, filename + '_model.sav', compress=3)
		
		print("Model: \n\t" + model_name + "\nSaved in:\n\t" + modelDir)
		
		summary = json.load(open(summaryDir, 'r'))
		summary[model_target][model_name] = dict()
		summary[model_target][model_name] = model_type

		with open(summaryDir, 'w') as json_file:
			json_file.write(json.dumps(summary))
			json_file.close()

		print("Model included in summary")

	def predict_channels(self, test, device, model, features, params, model_type, model_name, prediction_name, plot_result = True, min_date = None, max_date = None, clean_na = True, clean_na_method = 'drop', target_raster = '1Min'):

		if model_type == 'LSTM':
			scalerX_predict = params['scalerX']
			scalery_predict = params['scalery']
			n_lags = params['n_lags']
			try:
				alpha_filter = params['alpha_filter']
			except:
				alpha_filter = None
		elif model_type == 'RF' or model_type == 'SVR':
			print ('No specifics for {} type'.format(model_type))
		elif model_type == 'OLS':
			try:
				alpha_filter = params['alpha_filter']
			except:
				alpha_filter = None

		n_train_periods = params['ratio_train']

		list_features = list()
		for item in features:
			if 'REF' != item[0]:
				list_features.append(item[1])
				if item[1] not in self.readings[test]['devices'][device]['data'].columns:
					print ('{} not in {}. Cannot predict using this model'.format(item[1], test))
					break

		print ('Preparing devices from test {}'.format(test))
		dataframeModel = self.readings[test]['devices'][device]['data'].loc[:, list_features]
		dataframeModel = dataframeModel.apply(pd.to_numeric,errors='coerce')   
		
		# Resample
		dataframeModel = dataframeModel.resample(target_raster).mean()
		
		# Remove na
		if clean_na:
			if clean_na_method == 'fill':
				dataframeModel = dataframeModel.fillna(method='bfill').fillna(method='ffill')
			elif clean_na_method == 'drop':
				dataframeModel = dataframeModel.dropna()
		
		if min_date != None:
			dataframeModel = dataframeModel[dataframeModel.index > min_date]
		if max_date != None:
			dataframeModel = dataframeModel[dataframeModel.index < max_date]

		indexModel = dataframeModel.index

		# List of features for later use
		feature_list = list(dataframeModel.columns)
		features_array = np.array(dataframeModel)

		if model_type == 'RF' or model_type == 'SVR':
			## Get model prediction
			dataframe = pd.DataFrame(model.predict(features_array), columns = ['prediction']).set_index(indexModel)
			dataframeModel = dataframeModel.combine_first(dataframe)
			self.readings[test]['devices'][device]['data'][prediction_name] = dataframeModel['prediction']
			print ('Channel {} prediction finished'.format(prediction_name))


		elif model_type == 'LSTM':
			# To fix
			test_X, index_pred, n_obs = prep_prediction_ML(dataframeModel, list_features_predict, n_lags, alpha_filter, scalerX_predict, verbose = True)
			prediction = predict_ML(model, test_X, n_lags, scalery_predict)
			dataframe = pd.DataFrame(prediction, columns = [prediction_name]).set_index(index_pred)
			readings[test]['devices'][device_name]['data'][prediction_name] = dataframe.loc[:,prediction_name]
		
		elif model_type == 'OLS':

			if self.readings[test]['models'][model_name]['formula']:
				# Rename to formula
				for item in features:
					dataframeModel.rename(columns={item[1]: item[0]}, inplace=True)

			## Predict the model results
			datapredict, _ = prep_data_OLS(dataframeModel, features, 1)
			prediction = predict_OLS(model, datapredict, False, False, 'test')

			self.readings[test]['devices'][device]['data'][prediction_name] = prediction

		if plot_result:
			# Plot
			fig = plot.figure(figsize=(15,10))
		
			# Fitted values
			plot.plot(self.readings[test]['devices'][device]['data'].index, \
					  self.readings[test]['devices'][device]['data'][prediction_name], 'g', alpha = 0.5, label = 'Predicted value')
			plot.grid(True)
			plot.legend(loc='best')
			plot.title('Model prediction for {}'.format(prediction_name))
			plot.xlabel('Time (-)')
			plot.ylabel(prediction_name)
			plot.show()

	def calculateAlphaSense(self, reading_name, append_name, variables, options):
		# variables =  deltas, methods, overlapHours
		deltas = variables[0]
		methods = variables[1]
		overlapHours = variables[2]

		# methods[0][0] # CO classic
		# methods[0][1] # CO n/a

		# methods[1][0] # NO2 baseline
		# methods[1][1] # NO2 single_aux

		# methods[2][0] # OX baseline
		# methods[2][1] # OX single_aux

		# Check if we have a reference first in the dataset
		for kit in self.readings[reading_name]['devices']:
			if 'is_reference' in self.readings[reading_name]['devices'][kit]:
				print ('Reference found')
				refAvail = True
				dataframeRef = self.readings[reading_name]['devices'][kit]['data']
				break
			else:
				refAvail = False
				dataframeRef = ''

		# For each kit in the requested reading, calculate the pollutants
		for kit in self.readings[reading_name]['devices']:
			if 'alphasense' in self.readings[reading_name]['devices'][kit]:
				print ('Calculating test {} for kit {}. Appending {}'.format(reading_name, kit, append_name))
				
				sensorIDs = self.readings[reading_name]['devices'][kit]['alphasense']
				sensorID_CO = self.readings[reading_name]['devices'][kit]['alphasense']['CO']
				sensorID_NO2 = self.readings[reading_name]['devices'][kit]['alphasense']['NO2']
				sensorID_OX = self.readings[reading_name]['devices'][kit]['alphasense']['O3']
				sensorSlots = self.readings[reading_name]['devices'][kit]['alphasense']['slots']
							  
				sensorIDs = (['CO', sensorID_CO, methods[0][0], methods[0][1], sensorSlots.index('CO')+1, deltas[0]], 
							['NO2', sensorID_NO2, methods[1][0], methods[1][1], sensorSlots.index('NO2')+1, deltas[1]], 
							['O3', sensorID_OX, methods[2][0], methods[2][1], sensorSlots.index('O3')+1, deltas[2]])
									
				# Calculate correction
				self.readings[reading_name]['devices'][kit]['alphasense']['model_stats'] = dict()
				self.readings[reading_name]['ready_to_model'] = False
				self.readings[reading_name]['devices'][kit]['data'], self.readings[reading_name]['devices'][kit]['alphasense']['model_stats'][append_name] = calculatePollutantsAlpha(
						_dataframe = self.readings[reading_name]['devices'][kit]['data'], 
						_pollutantTuples = sensorIDs,
						_refAvail = refAvail, 
						_dataframeRef = dataframeRef, 
						_overlapHours = overlapHours, 
						_type_regress = 'best', 
						_filterExpSmoothing = filterExpSmoothing, 
						_trydecomp = options['checkBoxDecomp'],
						_plotsInter = options['checkBoxPlotsIn'], 
						_plotResult = options['checkBoxPlotsResult'],
						_verbose = options['checkBoxVerb'], 
						_printStats = options['checkBoxStats'],
						_calibrationDataPath = os.path.join(self.dataDirectory, 'interim/CalibrationData/'),
						_currentSensorNames = self.currentSensorNames,
						_append_name = append_name)

	def addChannelFormula(self, reading_name, device_name, new_channel_name, terms, formula):


		def functionFormula(reading_name, device_name, Aname, Bname, Cname, Dname, formula):
			# Create dataframe and merge everything
			calcData = pd.DataFrame()
			mergeData = pd.merge(pd.merge(pd.merge(self.readings[reading_name]['devices'][device_name]['data'].loc[:,(Aname,)],\
												   self.readings[reading_name]['devices'][device_name]['data'].loc[:,(Bname,)],\
												   left_index=True, right_index=True), \
										  self.readings[reading_name]['devices'][device_name]['data'].loc[:,(Cname,)], \
										  left_index=True, right_index=True),\
								 self.readings[reading_name]['devices'][device_name]['data'].loc[:,(Dname,)],\
								 left_index=True, right_index=True)
			# Assign names to columns
			calcData[Aname] = mergeData.iloc[:,0] #A
			calcData[Bname] = mergeData.iloc[:,1] #B
			calcData[Cname] = mergeData.iloc[:,2] #C
			calcData[Dname] = mergeData.iloc[:,3] #D
			A = calcData[Aname]
			B = calcData[Bname]
			C = calcData[Cname]
			D = calcData[Dname]
			# Eval the formula
			result = eval(formula)
			return result

		self.readings[reading_name]['devices'][device_name]['data'][new_channel_name] = functionFormula(reading_name, device_name,terms[0],terms[1], terms[2],terms[3], formula)    
		self.readings[reading_name]['ready_to_model'] = False
		print()

	def export_data(self, reading_name, device_export, export_path, to_processed_folder, processed_only, rename):


		def exportFile(_savePath, _name, _df):

			if not os.path.exists(_savePath):
				os.mkdir(_savePath)

			if not os.path.exists(_savePath + '/' + _name + '.csv'):
				_df.to_csv(_savePath + '/' + _name + '.csv', sep=",")
				print ('\tFile saved to: \n' + _savePath + '/' + _name +  '.csv')
			else:
				print("\tFile Already exists - delete it first, I don't want to overwrite anything!")


		df = self.readings[reading_name]['devices'][device_export]['data'].copy()

		with open(join(self.interimDirectory, 'sensorNamesExport.json')) as handle:
			sensorsDict = json.loads(handle.read())

		sensorShortTitles = list()
		sensorExportNames = list()
		sensorExportMask = list()

		for sensor in sensorsDict.keys():
			sensorShortTitles.append(sensorsDict[sensor]['shortTitle'])
			sensorExportNames.append(sensorsDict[sensor]['exportName'])
			if processed_only and sensorsDict[sensor]['processed'] != 'raw': sensorExportMask.append(True)
			elif processed_only and sensorsDict[sensor]['processed'] == 'raw': sensorExportMask.append(False)
			elif not processed_only: sensorExportMask.append(True)

		channels = list()

		for sensor in sensorShortTitles:
			if sensorExportMask[sensorShortTitles.index(sensor)]:

				if any(sensor == column for column in df.columns): exactMatch = True
				else: exactMatch = False
				
				for column in df.columns:
					if sensor in column and not exactMatch:
						if rename:
							df.rename(columns={column: sensorExportNames[sensorShortTitles.index(sensor)]}, inplace=True)
							channels.append(sensorExportNames[sensorShortTitles.index(sensor)])
						else:
							channels.append(column)
						break
					elif sensor == column and exactMatch:
						if rename:
							df.rename(columns={column: sensorExportNames[sensorShortTitles.index(sensor)]}, inplace=True)
							channels.append(sensorExportNames[sensorShortTitles.index(sensor)])
						else:
							channels.append(column)
						break
		print ('Exporting channels: \n {}'.format(channels))
		df = df.loc[:, channels]

		exportFile(export_path, device_export, df)
		
		if to_processed_folder:
			year = reading_name[0:4]
			month = reading_name[5:7]
			exportDir = os.path.join(self.dataDirectory, 'processed', year, month, reading_name, 'processed')
			print ('\tSaving files to: \n{}'.format(exportDir))
			exportFile(exportDir, device_export,  df)