from src.models.baseline_tools import *
from src.data.api import *	
from src.models.formulas import *	
from src.saf import *
from src.data.test import test_wrapper

class data_wrapper (saf):

	def __init__(self, verbose = True):
		try:
			saf.__init__(self, verbose)
			self.tests = dict()
			self.name_combined_data = self.config['data']['NAME_COMBINED_DATA']	

		except:
			traceback.print_exc()
		else:
			self.std_out('Test initialisation done', 'SUCCESS')

	def get_tests(self, directory):
		# Get available tests in the data folder structure
		tests = dict()
		mydir = join(directory, 'processed')
		for root, dirs, files in os.walk(mydir):
			for _file in files:
				if _file.endswith(".yaml"):
					filePath = join(root, _file)
					stream = open(filePath)
					yamlFile = yaml.load(stream, Loader = yaml.BaseLoader)
					tests[yamlFile['id']] = root
					
		return tests

	def available_tests(self):
		self.availableTests = self.get_tests(self.dataDirectory)
		return self.availableTests
	
	def preview_test(self, testPath):
		# Find Yaml
		filePath = join(testPath, 'test_description.yaml')
		with open(filePath, 'r') as stream:
			test = yaml.load(stream)
		
		test_id = test['id']
		
		self.std_out('Test Preview')

		self.std_out('Loading test {}'.format(test_id))
		self.std_out(test['comment'])
	
	def load_test(self, test_name, options = dict()):

		test = test_wrapper(test_name, self)
		test.load(options)
		
		self.tests[test_name] = test
		self.std_out('Test loaded successfully', 'SUCCESS', force = True)

	def load_devices_API(self, test_name, source_ids, options):
		from src.data.device import device_wrapper

		# Load data from the API
		self.tests[test_name] = test_wrapper(test_name, self)

		self.tests[test_name].add_details(dict())
		for device_id in source_ids:
			self.tests[test_name].devices[device_id] = device_wrapper({'device_id': device_id,
													  'frequency': options['frequency'],
                                           			  'type': None,
                                           			  'source': 'api'}, self)

			self.tests[test_name].devices[device_id].load(options = options)
		
		# # Case for merged API to CSV
		# else:
		# 	for key in data['devices'].keys():
		# 		self.tests[test_name]['devices'][key] = data['devices'][key] 

	def unload_test(self, test_name):
		if test_name in self.tests.keys():
			self.tests.pop(test_name)
		self.std_out('Unloading {}'.format(test_name))

	# # Temporary
	# def preprocess_test(self, test_name, window = 10):
	# 	for device in self.tests[test_name]['devices'].keys():
	# 		self.std_out(f'Preprocessing {device}', force = True)
	# 		self.tests[test_name]['devices'][device]['data_preprocess'] = self.tests[test_name]['devices'][device]['data'].rolling(window = window).mean()
	# 	self.std_out('Preprocessing done')

	def clear_tests(self):
		self.tests.clear()
		self.std_out('Clearing tests')

	# def describe_test(self, test_name, devices = None, verbose = True, tablefmt = 'simple'):
	# 	if test_name in self.tests.keys():
	# 		summary_dict = dict()
	# 		summary_dict[' '] = ['Min Date', 'Max Date',  'Total time delta (minutes)', 'Total time delta (days)','Number of records after drop (minutes)', 'Ratio (%)']

	# 		if devices is None: listDevices = self.tests[test_name]['devices'].keys()
	# 		else: listDevices = devices
				
	# 		for device in listDevices:
	# 			summary_dict[device] = list()
				
	# 			# print (f'Test: {testName}, device: {device}')
	# 			df = self.tests[test_name]['devices'][device]['data'].copy()
	# 			if len(df.index) > 0:
	# 				summary_dict[device].append(df.index[0])
	# 				summary_dict[device].append(df.index[-1])
	# 				summary_dict[device].append((df.index[-1]-df.index[0]).total_seconds()/60)
	# 				summary_dict[device].append((df.index[-1]-df.index[0]).total_seconds()/(3600*24))
	# 				df = df.resample('1Min').mean()
	# 				df.dropna(axis = 0, how='any', inplace=True)
	# 				summary_dict[device].append(len(df.index))
	# 				summary_dict[device].append(min(100,summary_dict[device][-1]/summary_dict[device][2]*100))
	# 			else:
	# 				summary_dict[device] = [None, None]
	# 		self.std_out(tabulate(summary_dict, numalign='right', headers="keys", tablefmt=tablefmt), force = verbose)

	# 		return summary_dict
	# 	else:
	# 		self.std_out(f'Reading {test_name} not loaded')

	def combine_devices(self, test_name):
		from src.data.device import device_wrapper
		
		try: 
			self.std_out(f'Combining devices for {test_name}')
			if self.name_combined_data in self.tests[test_name].devices.keys(): self.tests[test_name].devices.pop(self.name_combined_data, None)
			ignore_keys = []
			
			if 'models' in vars(self.tests[test_name]).keys():
				ignore_keys = self.tests[test_name].models.keys()

			if ignore_keys != []: self.std_out('Ignoring keys {}'.format(ignore_keys))

			dataframe_result = pd.DataFrame()
			## And then add it again
			for device in self.tests[test_name].devices.keys():
				if device not in ignore_keys:

					append = self.tests[test_name].devices[device].name
					new_names = list()
					for name in self.tests[test_name].devices[device].readings.columns:
						# print name
						new_names.append(name + '_' + append)
					
					self.tests[test_name].devices[device].readings.columns = new_names
					dataframe_result = dataframe_result.combine_first(self.tests[test_name].devices[device].readings)

			self.tests[test_name].devices[self.name_combined_data] = device_wrapper({'name': self.name_combined_data,
													  								'frequency': '1Min',
								                                           			 'type': '',
								                                           			 'source': ''}, self)

			self.tests[test_name].devices[self.name_combined_data].readings = dataframe_result
		except:
			self.std_out('Error ocurred while combining data. Review data', 'ERROR')
			traceback.print_exc()
			return False
		else:
			self.std_out('Data combined successfully', 'SUCCESS')
			return True

	def prepare_dataframe_model(self, model_object):

		test_names = list(model_object.data['train'].keys())

		# Create structure for multiple training
		if len(test_names) > 1: 
			multiple_training = True
			combined_name = model_object.name + '_' + self.config['models']['NAME_MULTIPLE_TRAINING_DATA']
			self.tests[combined_name] = dict()
			self.tests[combined_name].models[model_object.name] = dict()
		else:
			multiple_training = False

		for test_name in test_names:

			device = model_object.data['train'][test_name]['devices']
			reference = model_object.data['train'][test_name]['reference_device']

			self.std_out('Preparing dataframe model for test {}'.format(test_name))
		
			if self.combine_tests(test_name):

				## Send only the needed features
				list_features = list()
				list_features_multiple = list()
				features = model_object.data['features']
				try:
					# Create the list of features needed
					for item in features.keys():

						# Dirty horrible workaround
						if type(device) == list: device = device[0]

						if item == 'REF': 
							feature_name = features[item] + '_' + reference
							feature_name_multiple = features[item]
							reference_name = feature_name
							reference_name_multiple = feature_name_multiple
						else: 
							feature_name = features[item] + '_' + device
							feature_name_multiple = features[item]
						list_features.append(feature_name)
						list_features_multiple.append(feature_name_multiple)
					
					# Get features from data only and pre-process non-numeric data
					dataframeModel = self.tests[test_name].devices[self.name_combined_data].readings.loc[:,list_features]
					# Remove device names if multiple training
					if multiple_training:
						for i in range(len(list_features)):
							dataframeModel.rename(columns={list_features[i]: list_features_multiple[i]}, inplace=True)
					
					dataframeModel = dataframeModel.apply(pd.to_numeric, errors='coerce')   

					# Resample
					dataframeModel = dataframeModel.resample(model_object.options['frequency'], limit = 1).mean()
					# Remove na
					if model_object.options['clean_na']:
						
						if model_object.options['clean_na_method'] == 'fill':
							dataframeModel = dataframeModel.fillna(method='ffill')
						
						elif model_object.options['clean_na_method'] == 'drop':
							dataframeModel.dropna(axis = 0, how='any', inplace = True)

					if model_object.options['min_date'] is not None:
						dataframeModel = dataframeModel[dataframeModel.index > model_object.options['min_date']]
					if model_object.options['max_date'] is not None:
						dataframeModel = dataframeModel[dataframeModel.index < model_object.options['max_date']]

					if model_object.name is not None:
						# Create model_name entry
						self.tests[test_name].models[model_object.name]=dict()

						self.tests[test_name].models[model_object.name]['data'] = dataframeModel
						self.tests[test_name].models[model_object.name]['features'] = features
						self.tests[test_name].models[model_object.name]['reference'] = reference_name
						# Set flag
						self.tests[test_name].ready_to_model = True
					
				except:
					self.std_out(f'Dataframe model failed for {test_name}')
					traceback.print_exc()
					return None
				else: 
					self.std_out(f'Dataframe model generated successfully for {test_name}')
					
		if multiple_training:
			self.std_out('Multiple training datasets requested. Combining')
			# Combine everything
			frames = list()

			for test_name in test_names:
				frames.append(self.tests[test_name].models[model_object.name]['data'])

			self.tests[combined_name].models[model_object.name]['data'] = pd.concat(frames)
			self.tests[combined_name].models[model_object.name]['features'] = features
			self.tests[combined_name].models[model_object.name]['reference'] = reference_name_multiple

			return combined_name
		else:
			return test_name

	def archive_model(self, test_name, model_object, dataframe = None):
		try:
			# Model saving in previous entry
			self.tests[test_name].models[model_object.name]['model_object'] = model_object
			
			# Dataframe
			if dataframe is not None: self.tests[test_name].models[model_object.name]['data'] = dataframe
			
		except:
			self.std_out('Problem occured while archiving model', 'ERROR')
			traceback.print_exc()
			pass
		else:
			self.std_out('Model archived correctly', 'SUCCESS')

	def calculate_alphasense(self, test_name, variables, options, use_preprocessed = False):

		# Check if we have a reference first in the dataset
		for device in self.tests[test_name].devices.keys():
			if self.tests[test_name].devices[device].type == 'ANALYSER':
				self.std_out(f'Reference found: {device}')
				refAvail = True
				dataframeRef = self.tests[test_name].devices[device].readings
				break
			else:
				refAvail = False
				dataframeRef = ''

		# For each kit in the requested reading, calculate the pollutants
		for device in self.tests[test_name].devices.keys():

			if 'alphasense' in vars(self.tests[test_name].devices[device]).keys():
				self.std_out('Calculating test {} for kit {}'.format(test_name, device), force = True)
				
				 # Get sensor information
				sensorSlots = self.tests[test_name].devices[device].alphasense['slots']

				sensorIDs = dict()
				for pollutant in variables.keys():
					sensorSerialNumber = self.tests[test_name].devices[device].alphasense[pollutant]
					sensorIDs[pollutant] = [sensorSerialNumber, sensorSlots.index(pollutant)+1]
					
				# Calculate correction
				self.tests[test_name].devices[device].alphasense['model_stats'] = dict()
				self.tests[test_name].ready_to_model = False

				if use_preprocessed: _data = 'data_preprocess'
				else: _data = 'data'

				self.tests[test_name].devices[device].readings, correlationMetrics = calculatePollutantsAlpha(
						_dataframe = self.tests[test_name].devices[device].readings, 
						_sensorIDs = sensorIDs,
						_variables = variables,
						_refAvail = refAvail, 
						_dataframeRef = dataframeRef,  
						_type_regress = 'best', 
						_filterExpSmoothing = filterExpSmoothing, 
						_plotsInter = options['checkBoxPlotsIn'], 
						_plotResult = options['checkBoxPlotsResult'],
						_verbose = options['checkBoxVerb'], 
						_printStats = options['checkBoxStats'],
						_calibrationDataPath = join(self.dataDirectory, 'interim/CalibrationData/'),
						_currentSensorNames = self.currentSensorNames)
				self.tests[test_name].devices[device].alphasense['model_stats'].update(correlationMetrics)

		self.std_out('Calculation of test {} finished'.format(test_name), force = True)

	def add_channel_from_formula(self, test_name, device_name, new_channel_name, terms, formula):

		def function_formula(test_name, device_name, Aname, Bname, Cname, Dname, formula):
			# Create dataframe and merge everything
			calcData = pd.DataFrame()
			mergeData = pd.merge(pd.merge(pd.merge(self.tests[test_name].devices[device_name].readings.loc[:,(Aname,)],\
												   self.tests[test_name].devices[device_name].readings.loc[:,(Bname,)],\
												   left_index=True, right_index=True), \
										  self.tests[test_name].devices[device_name].readings.loc[:,(Cname,)], \
										  left_index=True, right_index=True),\
								 self.tests[test_name].devices[device_name].loc[:,(Dname,)],\
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

		self.tests[test_name].devices[device_name].readings[new_channel_name] = function_formula(test_name, device_name,terms[0],terms[1], terms[2],terms[3], formula)    
		self.tests[test_name].ready_to_model = False

	def export_data(self, test_name, device, export_path = '', to_processed_folder = False, all_channels = True, include_raw = False, include_processed = False, rename = False, forced_overwrite = False):

		df = self.tests[test_name].devices[device].readings.copy()
		if not all_channels:

			with open(join(self.interimDirectory, 'sensorNamesExport.json')) as handle:
				sensorsDict = json.loads(handle.read())

			sensorShortTitles = list()
			sensorExportNames = list()
			sensorExportMask = list()

			for sensor in sensorsDict.keys():
				sensorShortTitles.append(sensorsDict[sensor]['shortTitle'])
				sensorExportNames.append(sensorsDict[sensor]['exportName'])
				# Describe all cases for clarity ('na' are both, processed and raw)
				if include_processed and sensorsDict[sensor]['processed'] == 'processed': sensorExportMask.append(True)
				elif include_processed and sensorsDict[sensor]['processed'] == 'na': sensorExportMask.append(True)
				elif include_raw and sensorsDict[sensor]['processed'] == 'na': sensorExportMask.append(True)
				elif include_raw and sensorsDict[sensor]['processed'] == 'raw': sensorExportMask.append(True)
				else: sensorExportMask.append(False)
			channels = list()

			for sensor in sensorShortTitles:
				if sensorExportMask[sensorShortTitles.index(sensor)]:

					if any(sensor == column for column in df.columns): exactMatch = True
					else: exactMatch = False
					
					for column in df.columns:
						if sensor in column and not exactMatch and column not in channels:

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
			self.std_out('Exporting channels: \n {}'.format(channels))
			df = df.loc[:, channels]

		if export_path != '': self.export_CSV_file(export_path, device, df, forced_overwrite = forced_overwrite)
		
		if to_processed_folder:
			year = test_name[0:4]
			month = test_name[5:7]
			exportDir = join(self.dataDirectory, 'processed', year, month, test_name, 'processed')
			self.std_out('Saving files to: \n{}'.format(exportDir))
			self.export_CSV_file(exportDir, device,  df, forced_overwrite = forced_overwrite)