from test_utils import *
from api_utils import *
from os.path import join
from sklearn.externals import joblib
import json

class recordings:
	def __init__(self):
		self.readings = dict()

	def add_recording_CSV(self, reading_name, source_id, target_raster = '1Min', clean_na = True, clean_na_method = 'fill'):
		data = loadTest(source_id, target_raster, clean_na, clean_na_method)
		self.readings[reading_name] = dict()
		self.readings[reading_name] = data[reading_name]

		# Set flag
		self.readings[reading_name]['ready_to_model'] = False

	def add_recording_API(self, reading_name, source_id, min_date, max_date, target_raster = '1Min'):
		data = getReadingsAPI(source_id, target_raster, min_date, max_date)
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
			
			## And then add it again
			dataframe = combine_data(self.readings[reading_name]['devices'], True)
			self.readings[reading_name]['devices'][name_combined_data] = dict()
			self.readings[reading_name]['devices'][name_combined_data]['data'] = dict()
			self.readings[reading_name]['devices'][name_combined_data]['data'] = dataframe
			
			## Create dict for model comparison
			self.readings[reading_name]['models'] = dict()

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
				# Check for weird things in the data
				dataframeModel = dataframeModel.apply(pd.to_numeric,errors='coerce')   
				
				# Resample
				dataframeModel = dataframeModel.resample(target_raster).mean()
				
				# Remove na
				if clean_na:
					if clean_na_method == 'fill':
						dataframeModel = dataframeModel.fillna(method='bfill').fillna(method='ffill')
					elif clean_na_method == 'drop':
						dataframeModel = dataframeModel.dropna()
				dataframeModel = dataframeModel[dataframeModel.index > min_date]
				dataframeModel = dataframeModel[dataframeModel.index < max_date]
				
				if 'models' not in self.readings[reading_name].keys():
					self.readings[reading_name]['models']=dict()

				self.readings[reading_name]['models'][model_name]=dict()
				self.readings[reading_name]['models'][model_name]['data'] = dataframeModel
				self.readings[reading_name]['models'][model_name]['features'] = tuple_features

				for item in tuple_features:
					if item[0] == 'REF': 
						reference = dataframeModel.loc[:,item[1] + '_' + item[2]]
						reference_name = reference.name
				self.readings[reading_name]['models'][model_name]['reference'] = reference_name
				
				# Set flag
				self.readings[reading_name]['ready_to_model'] = True
				
			except:
				if item_name not in self.readings[reading_name]['devices'][name_combined_data]['data'].columns:
					print ('{} not in {}'.format(item_name, reading_name))
			else: 
				print ('\tDataframe model generated successfully')

	def archive_model(self, reading_name, model_name, metrics_model, dataframe, model, model_type, model_target, ratio_train, formula = '', n_lags = None, scalerX = None, scalery = None):
		try:
			# Metrics
			self.readings[reading_name]['models'][model_name]['metrics'] = metrics_model
	
			# Model and modelType
			self.readings[reading_name]['models'][model_name]['model'] = model
			self.readings[reading_name]['models'][model_name]['model_type'] = model_type
			self.readings[reading_name]['models'][model_name]['model_target'] = model_target
			if formula != '':
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
			# Dataframe
			self.readings[reading_name]['devices'][model_name] = dict()
			self.readings[reading_name]['devices'][model_name]['data'] = dataframe
		
		except:
			print ('Problem occured')
			pass
		else:
			print('Model archived correctly')

	def export_model(self, reading_name, model_name, model_directory):
		model_target = self.readings[reading_name]['models'][model_name]['model_target']
		model_type = self.readings[reading_name]['models'][model_name]['model_type']
		model = self.readings[reading_name]['models'][model_name]['model']

		modelDir = join(model_directory, 'Models/', model_target)
		summaryDir = join(model_directory, 'Models/summary.json')
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