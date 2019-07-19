from src.data.api_tools import *	
from os.path import join
from sklearn.externals import joblib
import json

import pandas as pd
import numpy as np

## Initialise paths and working directories
from os import pardir, getcwd
from os.path import join, abspath, normpath, basename
from src.data.test_tools import getSensorNames, getTests
from src.models.baseline_tools import *

import traceback

class recording:

	def __init__(self, verbose = True):
		
		self.readings = dict()
		self.rootDirectory = abspath(abspath(join(getcwd(), pardir)))
		self.dataDirectory = join(self.rootDirectory, 'data')
		self.interimDirectory = join(self.dataDirectory, 'interim')
		self.modelDirectory = join(self.rootDirectory, 'models')
		currentSensorsh = ('https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/2.1-dev/lib/Sensors/Sensors.h')
		self.availableTests = getTests(self.dataDirectory)
		self.verbose = verbose
		self.currentSensorNames = getSensorNames(currentSensorsh, join(self.dataDirectory, 'interim'))
		self.name_combined_data = 'COMBINED_DEVICES'

	def available_tests(self):
		self.availableTests = getTests(self.dataDirectory)
		return self.availableTests

	def std_out(self, msg):
		if self.verbose: print(msg)	
	
	def load_recording_database(self, reading_name, source_id, target_raster = '1Min', clean_na = True, clean_na_method = 'fill', load_processed = False, load_cached_API = True, cache_API = True):
		
		def loadTest(testPath, target_raster, currentSensorNames, clean_na = True, clean_na_method = 'fill', dataDirectory = '', load_processed = True, load_cached_API = True, cache_API = True):

			# Initialise dict for readings
			_readings = dict()

			# Find Yaml
			filePath = join(testPath, 'test_description.yaml')
			with open(filePath, 'r') as stream:
				test = yaml.load(stream)
			
			test_id = test['test']['id']
				
			_readings[test_id] = dict()
			_readings[test_id]['devices'] = dict()
			
			self.std_out('Test Load')

			self.std_out('Loading test {}'.format(test_id))
			self.std_out(test['test']['comment'])

			# Retrieve cached information, if any
			try:
				with open(join(testPath, 'cached', 'cache_info.json')) as handle:
					cached_info = json.loads(handle.read())
			except:
				cached_info = dict()

			# Open all kits
			for kit in test['test']['devices']['kits'].keys():
				self.std_out('Device ID {}'.format(kit))
				# Assume that if we don't specify the source, it's a csv (old)
				if 'source' not in test['test']['devices']['kits'][kit]:
					format_csv = True
				elif 'csv' in test['test']['devices']['kits'][kit]['source']:
					format_csv = True
				elif 'api' in test['test']['devices']['kits'][kit]['source']:
					format_csv = False
				
				if format_csv:
					
					try:
						metadata = test['test']['devices']['kits'][kit]['metadata']
					except:
						traceback.print_exc()
						metadata = ''
						pass
					
					# List for names conversion
					targetSensorNames = list()
					testSensorNames = list()

					for item_test in metadata:
						id_test = metadata[item_test]['id']

						for item_target in currentSensorNames:
							if currentSensorNames[item_target]['id'] == id_test and id_test != '0' and item_test not in testSensorNames:
								targetSensorNames.append(currentSensorNames[item_target]['shortTitle'])
								testSensorNames.append(item_test)            
					
					# Get fileName
					fileNameProc = test['test']['devices']['kits'][kit]['fileNameProc']
					
					filePath = join(testPath, fileNameProc)
					location = test['test']['devices']['kits'][kit]['location']
					self.std_out('Kit {} located {}'.format(kit, location))
					
					# Create dict for kit
					kitDict = dict()
					kitDict['location'] = location
					kitDict['data'] = readDataframeCsv(filePath, location, 
														target_raster, clean_na, clean_na_method, 
														targetSensorNames, testSensorNames)

					_readings[test_id]['devices'][kit] = kitDict
					
					## Check if it's a STATION and retrieve alphadelta codes
					if test['test']['devices']['kits'][kit]['type'] == 'STATION':
						alphasense = dict()
						alphasense['CO'] = test['test']['devices']['kits'][kit]['alphasense']['CO']
						alphasense['NO2'] = test['test']['devices']['kits'][kit]['alphasense']['NO2']
						alphasense['O3'] = test['test']['devices']['kits'][kit]['alphasense']['O3']
						alphasense['slots'] = test['test']['devices']['kits'][kit]['alphasense']['slots']
						_readings[test_id]['devices'][kit]['alphasense'] = alphasense
						self.std_out('ALPHASENSE')
						self.std_out(alphasense)
						
					self.std_out('Kit {} has been loaded'.format(kit))
				
				else:
					device_id = test['test']['devices']['kits'][kit]['device_id']
					list_devices_api = list()
					list_devices_api.append(device_id)
					
					if test['test']['devices']['kits'][kit]['min_date'] != None: 
						min_date = datetime.strptime(test['test']['devices']['kits'][kit]['min_date'], '%Y-%m-%d')
					else: 
						min_date = None
					
					if test['test']['devices']['kits'][kit]['max_date'] != None: 
						max_date=datetime.strptime(test['test']['devices']['kits'][kit]['max_date'], '%Y-%m-%d')
					else:
						max_date = None
					# Flag to combine cached data and API data if there is newer
					combine_cache_API = False
					if load_cached_API:
						try:
							# Load cached data here
							fileName= join(testPath, 'cached', kit  + '.csv')

							# Create dict for kit
							kitDict = dict()
							location = cached_info[kit]['location']
							alphasense = cached_info[kit]['alphasense']
							kitDict['location'] = location
							kitDict['data'] = readDataframeCsv(fileName, location, target_raster, clean_na, clean_na_method)
							kitDict['alphasense'] = alphasense
							_readings[test_id]['devices'][kit] = kitDict

						except:
							self.std_out('Problem loading cached data, requesting to API')
							traceback.print_exc()
							load_API = True
						
						else:
							self.std_out('Loaded cached files from: \n{}'.format(fileName))

							last_reading_cached = _readings[test_id]['devices'][kit]['data'].index[-1]
							self.std_out('Last day in cached data {}'.format(last_reading_cached))
							last_reading_api = datetime.strptime(getDateLastReading(device_id), '%Y-%m-%dT%H:%M:%SZ')
							last_reading_api = pd.to_datetime(last_reading_api).tz_localize('UTC').tz_convert(location)
							self.std_out('Last reading in API {}'.format(last_reading_api))
							

							# Check which dates to load
							if max_date != None:
								# Localize max test date for comparison
								max_date_localized = pd.to_datetime(max_date).tz_localize('UTC').tz_convert(location)
								self.std_out('Max date in test {}'.format(max_date_localized))
								# Check what where we need to load data from, if any
								if last_reading_cached < max_date_localized and last_reading_api > last_reading_cached + timedelta(days=1):
									load_API = True
									combine_cache_API = True
									min_date = last_reading_cached
									max_date = min(max_date_localized, last_reading_api)
								else:
									load_API = False
							else:
								# If no test data specified, check the last reading in the API
								if last_reading_api > last_reading_cached + timedelta(days=1):
									load_API = True
									combine_cache_API = True
									min_date = last_reading_cached
									max_date = last_reading_api
								else:
									load_API = False
					
					else:
						load_API = True

					# Either we couldn't succeed getting cached data or we were forced to get the API data
					if load_API:
						self.std_out('Requesting device to API')

						# Get readings from the API
						data = getReadingsAPI(list_devices_api, target_raster, min_date, max_date, currentSensorNames, dataDirectory, clean_na, clean_na_method)
						location = data['devices'][device_id]['location']
						alphasense_data = None
						if 'alphasense' in data['devices'][kit].keys(): alphasense_data = data['devices'][kit]['alphasense']

						# Check if the kit name is the same as the platform name
						if kit not in data['devices'].keys():
							self.std_out('Device name in platform is not the same as test name, using test name')
							# List of sensors available
							list_keys = list(data['devices'])
							self.std_out ('Channel list {}'.format(list_keys))
							_readings[test_id]['devices'][kit] = data['devices'][list_keys[0]]
						else:
							self.std_out('Device name in platform is the same as test name')
							# Combine data if there is new data
							if combine_cache_API: _readings[test_id]['devices'][kit]['data'] = _readings[test_id]['devices'][kit]['data'].combine_first(data['devices'][kit]['data'])
							# Or just save it in readings
							else: _readings[test_id]['devices'][kit] = data['devices'][kit]
							self.std_out('New updated max date in test {}'.format(_readings[test_id]['devices'][kit]['data'].index[-1]))

					# Cache the files if requested
					if cache_API and load_API:
						self.std_out('Caching files for {}'.format(kit))

						# New data is cached for later use
						cached_info[kit] = dict()
						cached_info[kit]['location'] = location
						cached_info[kit]['alphasense'] = alphasense_data
						# New data wo process is cache-wide info
						cached_info['new_data_wo_process'] = load_API

						filePath = join(testPath, 'cached')
						self.exportCSVFile(filePath, kit, _readings[test_id]['devices'][kit]['data'], forced_overwrite = True)

						self.std_out('Dumping cache.info')
						with open(join(filePath, 'cache_info.json'), 'w') as file:
							json.dump(cached_info, file)

				# Load processed data with '_processed' appendix
				if load_processed:
					kitDict_processed = dict()
					kitDict_processed['location'] = location
					filePath_processed = join(testPath, 'processed', kit + '.csv')
					if os.path.exists(filePath_processed):
						self.std_out('Found processed data. Loading...')
						kitDict_processed['data'] = readDataframeCsv(filePath_processed, location, target_raster, clean_na, clean_na_method, targetSensorNames, testSensorNames)
						_readings[test_id]['devices'][kit + '_processed'] = kitDict_processed


			## Check if there's was a reference equipment during the test
			if 'reference' in test['test']['devices'].keys():
					refAvail = True
			else:
				refAvail = False

			if refAvail:
				self.std_out('REFERENCE')
				for reference in test['test']['devices']['reference']:
					self.std_out(reference)
					referenceDict =  dict()
					
					# Get the file name and frequency
					fileNameProc = test['test']['devices']['reference'][reference]['fileNameProc']
					frequency_ref = test['test']['devices']['reference'][reference]['index']['frequency']
					if target_raster != frequency_ref:
						self.std_out('Resampling reference')

					# Check the index name
					timeIndex = test['test']['devices']['reference'][reference]['index']['name']
					location = test['test']['devices']['reference'][reference]['location']
					self.std_out('Reference location {}'.format(location))
					
					# Open it with pandas    
					filePath = join(testPath, fileNameProc)
					df = readDataframeCsv(filePath, location, target_raster, clean_na, clean_na_method, [], [], timeIndex)
					
					## Convert units
					# Get which pollutants are available in the reference
					pollutants = test['test']['devices']['reference'][reference]['channels']['pollutants']
					channels = test['test']['devices']['reference'][reference]['channels']['names']
					units = test['test']['devices']['reference'][reference]['channels']['units']
					
					for index in range(len(channels)):
						
						pollutant = pollutants[index]
						channel = channels[index]
						unit = units[index]
						
						# Get molecular weight and target units for the pollutant in question
						for pollutantItem in pollutantLUT:
							if pollutantItem[0] == pollutant:
								molecularWeight = pollutantItem[1]
								targetUnit = pollutantItem[2]
								
						convertionLUT = (['ppm', 'ppb', 1000],
							 ['mg/m3', 'ug/m3', 1000],
							 ['mg/m3', 'ppm', 24.45/molecularWeight],
							 ['ug/m3', 'ppb', 24.45/molecularWeight],
							 ['mg/m3', 'ppb', 1000*24.45/molecularWeight],
							 ['ug/m3', 'ppm', 1./1000*24.45/molecularWeight])
						
						# Get convertion factor
						if unit == targetUnit:
								convertionFactor = 1
								self.std_out('No unit convertion needed for {}'.format(pollutant))
						else:
							for convertionItem in convertionLUT:
								if convertionItem[0] == unit and convertionItem[1] == targetUnit:
									convertionFactor = convertionItem[2]
								elif convertionItem[1] == unit and convertionItem[0] == targetUnit:
									convertionFactor = 1.0/convertionItem[2]
							self.std_out('Converting {} from {} to {}'.format(pollutant, unit, targetUnit))
						
						df.loc[:,pollutant + '_' + ref_append] = df.loc[:,channel]*convertionFactor
						
					referenceDict['data'] = df
					_readings[test_id]['devices'][reference] = referenceDict
					_readings[test_id]['devices'][reference]['is_reference'] = True
					self.std_out('{} reference has been loaded'.format(reference))
				
			return _readings

		def readDataframeCsv(filePath, location, target_raster, clean_na, clean_na_method, targetNames = [], testNames = [], refIndex = 'Time'):
			# Create pandas dataframe
			df = pd.read_csv(filePath, verbose=False, skiprows=[1])

			if 'Time' in df.columns:
				df = df.set_index('Time')
			elif 'TIME' in df.columns:
				df = df.set_index('TIME')
			elif refIndex != '':
				df = df.set_index(refIndex)
			else:
				self.std_out('No known index found')

			# Set index
			df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)

			# Remove duplicates
			df = df[~df.index.duplicated(keep='first')]

			# Sort index
			df.sort_index(inplace=True)

			# Drop unnecessary columns
			df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
			
			# Check for weird things in the data
			df = df.apply(pd.to_numeric,errors='coerce')   
			
			# Resample
			df = df.resample(target_raster).mean()
			
			# Remove na
			if clean_na:
				if clean_na_method == 'fill':
					df = df.fillna(method='bfill').fillna(method='ffill')
				elif clean_na_method == 'drop':
					df = df.dropna()
			
			# Create dictionary and add it to the _readings key
			if len(targetNames) == len(testNames) and len(targetNames) > 0:
				for i in range(len(targetNames)):
					if not (testNames[i] == '') and not (testNames[i] == targetNames[i]) and testNames[i] in df.columns:
						df.rename(columns={testNames[i]: targetNames[i]}, inplace=True)
						self.std_out('Renaming column {} to {}'.format(testNames[i], targetNames[i]))

			return df

		data = loadTest(source_id, target_raster, self.currentSensorNames, clean_na, clean_na_method, self.dataDirectory, load_processed, load_cached_API, cache_API)
		self.readings[reading_name] = dict()
		self.readings[reading_name] = data[reading_name]

		# Set flag
		self.readings[reading_name]['ready_to_model'] = False

	def load_recording_API(self, reading_name, source_id, min_date, max_date, target_raster = '1Min', clean_na = True, clean_na_method = 'fill'):
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
		self.std_out('Deleting', reading_name)

	def clear_recordings(self):
		self.readings.clear()
		self.std_out('Clearing recordings')

	def combine_readings(self, reading_name):
		
		def combine_data(list_of_datas, ignore_keys = []):
			dataframe_result = pd.DataFrame()

			for i in list_of_datas:
				if i not in ignore_keys:

					dataframe = pd.DataFrame()
					dataframe = dataframe.combine_first(list_of_datas[i]['data'])

					append = i
					prepend = ''
					new_names = list()
					for name in dataframe.columns:
						# print name
						new_names.append(prepend + name + '_' + append)
					
					dataframe.columns = new_names
					dataframe_result = dataframe_result.combine_first(dataframe)

			return dataframe_result

		try: 
			## Since we don't know if there are more or less channels than last time
			## (and tbh, I don't feel like checking), remove the key
			self.readings[reading_name]['devices'].pop(self.name_combined_data, None)
			ignore_keys = []
			
			if 'models' in self.readings[reading_name].keys():
				ignore_keys = self.readings[reading_name]['models'].keys()

			if ignore_keys != []: self.std_out('Ignoring keys {}'.format(ignore_keys))
			## And then add it again
			dataframe = combine_data(self.readings[reading_name]['devices'], ignore_keys)

			self.readings[reading_name]['devices'][self.name_combined_data] = dict()
			self.readings[reading_name]['devices'][self.name_combined_data]['data'] = dict()
			self.readings[reading_name]['devices'][self.name_combined_data]['data'] = dataframe

		except:
			self.std_out('Error ocurred while combining data. Review data')
			traceback.print_exc()
			return False
		else:
			self.std_out('Data combined successfully')
			return True

	def prepare_dataframe_model(self, reading_name, features, device, reference, min_date, max_date, model_name, clean_na = True, clean_na_method = 'fill' , target_raster = '1Min'):
		self.std_out('Preparing dataframe model for test {}'.format(reading_name))
		if self.combine_readings(reading_name):

			## Send only the needed features
			list_features = list()
			try:
				# Create the list of features needed
				for item in features.keys():

					# Dirty horrible workaround
					if type(device) == list: device = device[0]
					
					if item == 'REF': 
						feature_name = features[item] + '_' + reference
						reference_name = feature_name
					else: feature_name = features[item] + '_' + device
					list_features.append(feature_name)
				
				# Get features from data only and pre-process non-numeric data
				dataframeModel = self.readings[reading_name]['devices'][self.name_combined_data]['data'].loc[:,list_features]
				dataframeModel = dataframeModel.apply(pd.to_numeric, errors='coerce')   

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
						self.std_out('Creating models session in recordings')
						self.readings[reading_name]['models']=dict()

					self.readings[reading_name]['models'][model_name]=dict()
					self.readings[reading_name]['models'][model_name]['data'] = dataframeModel
					self.readings[reading_name]['models'][model_name]['features'] = features

					# Put reference name
					self.readings[reading_name]['models'][model_name]['reference'] = reference_name
					self.std_out('Model reference is {}'.format(reference))

					# Set flag
					self.readings[reading_name]['ready_to_model'] = True
				
			except:
				self.std_out('Dataframe model failed')
				traceback.print_exc()
				return False
			else: 
				self.std_out('Dataframe model generated successfully')
				return True

	def archive_model(self, reading_name, model_name, model_type, metrics_model, hyperparameters, options, dataframe = None, model = None):
		try:
			# Metrics
			if metrics_model != None: self.readings[reading_name]['models'][model_name]['metrics'] = metrics_model
	
			# Model and modelType
			if model != None: self.readings[reading_name]['models'][model_name]['model'] = model
			
			self.readings[reading_name]['models'][model_name]['model_type'] = model_type
			self.readings[reading_name]['models'][model_name]['model_target'] = options['model_target']
			if 'formula' in options.keys(): self.readings[reading_name]['models'][model_name]['formula'] = options['formula']
	
			# Parameters
			self.readings[reading_name]['models'][model_name]['hyperparameters'] = hyperparameters
			
			# Dataframe
			if dataframe != None:
				self.readings[reading_name]['devices'][model_name] = dict()
				self.readings[reading_name]['devices'][model_name]['data'] = dataframe
			
		except:
			self.std_out('Problem occured while archiving model')
			traceback.print_exc()
			pass
		else:
			self.std_out('Model archived correctly')

	def calculateAlphaSense(self, reading_name, append_name, variables, options):

		# variables =  hyperparameters, methods, overlapHours
		hyperparameters = variables[0]
		methods = variables[1]
		overlapHours = variables[2]
		baseline_method = variables[3]

		# methods[0][0] # CO classic
		# methods[0][1] # CO n/a

		# methods[1][0] # NO2 baseline
		# methods[1][1] # NO2 single_aux

		# methods[2][0] # OX baseline
		# methods[2][1] # OX single_aux

		# Check if we have a reference first in the dataset
		for kit in self.readings[reading_name]['devices']:
			if 'is_reference' in self.readings[reading_name]['devices'][kit]:
				self.std_out('Reference found')
				refAvail = True
				dataframeRef = self.readings[reading_name]['devices'][kit]['data']
				break
			else:
				refAvail = False
				dataframeRef = ''

		# For each kit in the requested reading, calculate the pollutants
		for kit in self.readings[reading_name]['devices']:
			if 'alphasense' in self.readings[reading_name]['devices'][kit]:
				self.std_out('---------------------')
				self.std_out('Calculating test {} for kit {}. Appending {}'.format(reading_name, kit, append_name))
				
				sensorID_CO = self.readings[reading_name]['devices'][kit]['alphasense']['CO']
				sensorID_NO2 = self.readings[reading_name]['devices'][kit]['alphasense']['NO2']
				sensorID_OX = self.readings[reading_name]['devices'][kit]['alphasense']['O3']
				sensorSlots = self.readings[reading_name]['devices'][kit]['alphasense']['slots']

				sensorIDs = (['CO', sensorID_CO, methods['CO'][0], baseline_method, methods['CO'][1], sensorSlots.index('CO')+1, hyperparameters], 
							['NO2', sensorID_NO2, methods['NO2'][0], baseline_method, methods['NO2'][1], sensorSlots.index('NO2')+1, hyperparameters], 
							['O3', sensorID_OX, methods['O3'][0], baseline_method, methods['O3'][1], sensorSlots.index('O3')+1, hyperparameters])
					
				# Calculate correction
				self.readings[reading_name]['devices'][kit]['alphasense']['model_stats'] = dict()
				self.readings[reading_name]['ready_to_model'] = False
				self.readings[reading_name]['devices'][kit]['data'], self.readings[reading_name]['devices'][kit]['alphasense']['model_stats'][append_name] = calculatePollutantsAlpha(
						_dataframe = self.readings[reading_name]['devices'][kit]['data'], 
						_sensorIDs = sensorIDs,
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

	def exportCSVFile(self, savePath, name, df, forced_overwrite = False):

		# If path does not exist, create it
		if not os.path.exists(savePath):
			os.mkdir(savePath)

		# If file does not exist 
		if not os.path.exists(savePath + '/' + name + '.csv') or forced_overwrite:
			df.to_csv(savePath + '/' + name + '.csv', sep=",")
			self.std_out('File saved to: \n' + savePath + '/' + name +  '.csv')
		else:
			self.std_out("File Already exists - delete it first, I was not asked to overwrite anything!")

	def export_data(self, reading_name, device_export, export_path = '', to_processed_folder = False, all_channels = False, hide_raw = False, rename = False, forced_overwrite = False):

		df = self.readings[reading_name]['devices'][device_export]['data'].copy()

		if not all_channels:

			with open(join(self.interimDirectory, 'sensorNamesExport.json')) as handle:
				sensorsDict = json.loads(handle.read())

			sensorShortTitles = list()
			sensorExportNames = list()
			sensorExportMask = list()

			for sensor in sensorsDict.keys():
				sensorShortTitles.append(sensorsDict[sensor]['shortTitle'])
				sensorExportNames.append(sensorsDict[sensor]['exportName'])
				if hide_raw and sensorsDict[sensor]['processed'] != 'raw': sensorExportMask.append(True)
				elif hide_raw and sensorsDict[sensor]['processed'] == 'raw': sensorExportMask.append(False)
				elif not hide_raw: sensorExportMask.append(True)

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

		if export_path != '': self.exportCSVFile(export_path, device_export, df, forced_overwrite = forced_overwrite)
		
		if to_processed_folder:
			year = reading_name[0:4]
			month = reading_name[5:7]
			exportDir = join(self.dataDirectory, 'processed', year, month, reading_name, 'processed')
			self.std_out('Saving files to: \n{}'.format(exportDir))
			self.exportCSVFile(exportDir, device_export,  df, forced_overwrite = forced_overwrite)