from src.saf import std_out
from src.saf import paths, configuration
from src.data.test import Test
from traceback import print_exc
from os import walk
from os.path import join, exists
import yaml
from time import strftime, localtime

class Data(object):

	def __init__(self, verbose = True):
		try:
			self.tests = dict()
			self.models = dict()
		except:
			print_exc()
		else:
			std_out('Data initialisation done', 'SUCCESS')

	
	def get_tests(self, directory, deep_description = False):
		'''
			Gets the tests in the given directory, looking for test_description.yaml
		'''

		# Get available tests in the data folder structure
		tests = dict()
		tdir = join(directory, 'processed')
		for root, dirs, files in walk(tdir):
			for file in files:
				if file.endswith(".yaml"):
					filePath = join(root, file)
					stream = open(filePath)
					yamlFile = yaml.load(stream, Loader = yaml.FullLoader)
					
					if deep_description == True:
						tests[yamlFile['id']] = dict()
						tests[yamlFile['id']]['path'] = root
						for key in yamlFile.keys():
							if key == 'devices': continue
							tests[yamlFile['id']][key] = yamlFile[key]
					else:
						tests[yamlFile['id']] = root

		return tests

	def available_tests(self):
		self.availableTests = self.get_tests(paths['dataDirectory'])
		return self.availableTests

	def tests_summary(self):
		return self.get_tests(paths['dataDirectory'], deep_description = True)

	def describe_test(self, test_name, devices = None, verbose = True, tablefmt = 'simple'):
        # TODO: make it cleaner
		if test_name in self.tests.keys():
			summary_dict = dict()
			summary_dict[' '] = ['Min Date', 'Max Date',  'Total time delta (minutes)', 'Total time delta (days)', 'Number of records after drop (minutes)', 'Ratio (%)']

			if devices is None: listDevices = self.tests[test_name].devices.keys()
			else: listDevices = devices
				
			for device in listDevices:
				summary_dict[device] = list()
				
				# print (f'Test: {testName}, device: {device}')
				df = self.tests[test_name].devices[device].readings.copy()
				if len(df.index) > 0:
					summary_dict[device].append(df.index[0])
					summary_dict[device].append(df.index[-1])
					summary_dict[device].append((df.index[-1]-df.index[0]).total_seconds()/60)
					summary_dict[device].append((df.index[-1]-df.index[0]).total_seconds()/(3600*24))
					df = df.resample('1Min').mean()
					df.dropna(axis = 0, how='any', inplace=True)
					summary_dict[device].append(len(df.index))
					summary_dict[device].append(min(100,summary_dict[device][-1]/summary_dict[device][2]*100))
				else:
					summary_dict[device] = [None, None]
			std_out(tabulate(summary_dict, numalign='right', headers="keys", tablefmt=tablefmt), force = True)

			return summary_dict
		else:
			std_out(f'Reading {test_name} not loaded')
	
	def preview_test(self, path):
		# Find Yaml
		filePath = join(path, 'test_description.yaml')

		with open(filePath, 'r') as stream:
			test = yaml.load(stream)
		
		test_id = test['id']
		
		std_out('Test preview')
		std_out(test['comment'])
	
	def load_test(self, test_name, options = dict()):
		
		std_out('Loading test {}'.format(test_name))
		test = Test(test_name)
		test.load(options)
		
		self.tests[test_name] = test
		
		if test_name in self.tests.keys(): 
			std_out('Test loaded successfully', 'SUCCESS', force = True)
			return True
		std_out(f'Test {test_name} not-loaded successfully', 'ERROR')	
		return False

	def load_devices(self, test_name, devices = list(), options = dict()):

		if type(devices) != list: 
			std_out(f'Devices should be list. Current type: {type(devices)}', 'ERROR')
			return False

		tname = strftime("%Y-%m", localtime()) + '_INT_' + test_name
		test = Test(tname)
		for device in devices:
			test.add_device(device)

		test.make()
		test.load(options)

		self.tests[tname] = test

		if tname in self.tests.keys(): 
			std_out('Test loaded successfully', 'SUCCESS', force = True)
			return True
		std_out(f'Test {tname} not-loaded successfully', 'ERROR')	
		return False		

	def unload_test(self, test_name):
		if test_name in self.tests.keys():
			self.tests.pop(test_name)
		
		if test_name not in self.tests.keys(): 
			std_out(f'Test {test_name} unloaded successfully', 'SUCCESS')
			return True
		std_out(f'Test {test_name} not-unloaded successfully', 'ERROR')	
		return False

	def process_test(self, test_name):
		return self.tests[test_name].process()

	def clear_tests(self):
		self.tests.clear()
		std_out('Tests cleared', 'SUCCESS')

	def export(self, test_name = None, path = '', forced_overwrite = False):
		# Make list for test names
		if test_name is None: tnames = list(self.tests.keys())
		else: tnames = [test_name]

		export_ok = True
		# Export each one of them
		for tname in tnames:
			# If path is empty, send to process folder of each test
			if path == '': epath = join(paths['dataDirectory'], 'processed', tname[0:4], tname[5:7], tname, 'processed')
			else: epath = path

			# Export to csv
			for device in self.tests[tname].devices.keys():
				export_ok &= self.tests[tname].devices[device].export(epath, forced_overwrite = forced_overwrite)

		if export_ok: std_out(f'Test {test_name} exported successfully', 'SUCCESS')
		else: std_out(f'Test {test_name} not exported successfully', 'ERROR')		
		return export_ok

	def combine_devices(self, test_name):
		from src.data.device import device_wrapper

		if test_name not in self.tests.keys(): 
			std_out(f'{test_name} is not loaded')
			return False
		
		try: 
			std_out(f'Combining devices for {test_name}')
			if configuration['data']['combined_devices_name'] in self.tests[test_name].devices.keys(): 
				self.tests[test_name].devices.pop(configuration['data']['combined_devices_name'], None)
			ignore_keys = []
			if 'models' in vars(self.tests[test_name]).keys():
				ignore_keys = self.tests[test_name].models.keys()

			if ignore_keys != []: std_out('Ignoring keys {}'.format(ignore_keys))

			dataframe_result = pd.DataFrame()
			## And then add it again
			for device in self.tests[test_name].devices.keys():
				if device not in ignore_keys:

					append = self.tests[test_name].devices[device].name
					new_names = list()
					for name in self.tests[test_name].devices[device].readings.columns:
						new_names.append(name + '_' + append)
					dataframe = self.tests[test_name].devices[device].readings.copy()
					dataframe.columns = new_names
					dataframe_result = dataframe_result.combine_first(dataframe)
			self.tests[test_name].devices[configuration['data']['combined_devices_name']] = device_wrapper({'name': configuration['data']['combined_devices_name'],
																					'frequency': '1Min',
																					 'type': '',
																					 'source': ''}, self)

			self.tests[test_name].devices[configuration['data']['combined_devices_name']].readings = dataframe_result
		except:
			std_out('Error ocurred while combining data. Review data', 'ERROR')
			print_exc()
			return False
		else:
			std_out('Data combined successfully', 'SUCCESS')
			return True

	def prepare_dataframe_model(self, model):

		test_names = list(model.data['train'].keys())

		# Create structure for multiple training
		if len(test_names) > 1: 
			multiple_training = True
			combined_name = model.name + '_' + configuration['models']['name_multiple_training_data']
			frames = list()
		else:
			multiple_training = False

		for test_name in test_names:

			device = model.data['train'][test_name]['devices']
			reference = model.data['train'][test_name]['reference_device']

			std_out('Preparing dataframe model for test {}'.format(test_name))
		
			if self.combine_devices(test_name):

				## Send only the needed features
				list_features = list()
				list_features_multiple = list()
				features = model.data['features']
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
					dataframeModel = self.tests[test_name].devices[configuration['data']['combined_devices_name']].readings.loc[:,list_features]
					
					# Remove device names if multiple training
					if multiple_training:
						for i in range(len(list_features)):
							dataframeModel.rename(columns={list_features[i]: list_features_multiple[i]}, inplace=True)

					dataframeModel = dataframeModel.apply(pd.to_numeric, errors='coerce')   

					# Resample
					dataframeModel = dataframeModel.resample(model.data['data_options']['frequency'], limit = 1).mean()
					
					# Remove na
					if model.data['data_options']['clean_na']:
						std_out('Cleaning na with {}'.format(model.data['data_options']['clean_na_method']))
						if model.data['data_options']['clean_na_method'] == 'fill':
							dataframeModel = dataframeModel.fillna(method='ffill')
						
						elif model.data['data_options']['clean_na_method'] == 'drop':
							dataframeModel.dropna(axis = 0, how='any', inplace = True)

					if model.data['data_options']['min_date'] is not None:
						dataframeModel = dataframeModel[dataframeModel.index > model.data['data_options']['min_date']]
					if model.data['data_options']['max_date'] is not None:
						dataframeModel = dataframeModel[dataframeModel.index < model.data['data_options']['max_date']]

					if multiple_training:
						frames.append(dataframeModel)
					
				except:
					std_out(f'Dataframe model failed for {test_name}')
					print_exc()
					return None
				else: 
					# Set flag
					self.tests[test_name].ready_to_model = True
					std_out(f'Dataframe model generated successfully for {test_name}')
					
		if multiple_training:
			std_out('Multiple training datasets requested. Combining')
			# Combine everything
			model.dataframe = pd.concat(frames)
			model.features = features		
			model.reference = reference_name_multiple

			return combined_name
		else:
			model.dataframe = dataframeModel
			model.features = features
			model.reference = reference_name
			
			return test_name

	def archive_model(self, model):
		try:
			# Model saving in previous entry
			self.models[model.name] = model
			
		except:
			std_out('Problem occured while archiving model', 'ERROR')
			print_exc()
			pass
		else:
			std_out('Model archived correctly', 'SUCCESS')

	# TODO - Remove from here
	def calculate_alphasense(self, test_name, variables, options, use_preprocessed = False):

		# Check if we have a reference first in the dataset
		for device in self.tests[test_name].devices.keys():
			if self.tests[test_name].devices[device].type == 'ANALYSER':
				std_out(f'Reference found: {device}')
				refAvail = True
				dataframeRef = self.tests[test_name].devices[device].readings
				break
			else:
				refAvail = False
				dataframeRef = ''

		# For each kit in the requested reading, calculate the pollutants
		for device in self.tests[test_name].devices.keys():
			if 'alphasense' in vars(self.tests[test_name].devices[device]).keys():
				std_out('Calculating test {} for kit {}'.format(test_name, device), force = True)
				
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
						_calibrationDataPath = join(self.interimDirectory, 'calibration/'),
						_currentSensorNames = self.currentSensorNames)
				self.tests[test_name].devices[device].alphasense['model_stats'].update(correlationMetrics)

		std_out('Calculation of test {} finished'.format(test_name), force = True)

	def upload_to_zenodo(self, upload_descriptor_name, sandbox = True, dry_run = True):
		'''
		This section uses the code inspired by this repo https://github.com/darvasd/upload-to-zenodo
		'''
		from src.secrets import ZENODO_TOKEN
		
		def fill_template(individual_descriptor, descriptor_file_name, upload_type = 'dataset'):
			# Open base template with all keys

			if upload_type == 'dataset': template_file_name = 'template_zenodo_dataset'
			elif upload_type == 'publication': template_file_name = 'template_zenodo_publication'

			with open (join(self.rootDirectory, f'src/data/{template_file_name}.json'), 'r') as template_file:
				template = json.load(template_file)

			filled_template = template

			# Fill it up for each key
			for key in individual_descriptor.keys():

				value = individual_descriptor[key]

				if key in filled_template['metadata'].keys():
					filled_template['metadata'][key] = value

			with open (join( paths['dataDirectory'], 'uploads', descriptor_file_name), 'w') as descriptor_json:
				json.dump(filled_template, descriptor_json, ensure_ascii=True)
				std_out(f'Created descriptor file for {descriptor_file_name}', 'SUCCESS')
			
			return json.dumps(filled_template)

		def get_submission_id(metadata, base_url):
			url = f"{base_url}/api/deposit/depositions"

			headers = {"Content-Type": "application/json"}
			response = requests.post(url, params={'access_token': ZENODO_TOKEN}, data=metadata, headers=headers)
			if response.status_code > 210:
				std_out("Error happened during submission, status code: " + str(response.status_code), 'ERROR')
				std_out(response.json(), "ERROR")
				return None

			# Get the submission ID
			submission_id = json.loads(response.text)["id"]

			return submission_id

		def upload_file(url, upload_metadata, files):
			response = requests.post(url, params={'access_token': ZENODO_TOKEN}, data = upload_metadata, files=files)
			return response.status_code		

		
		std_out(f'Uploading {upload_descriptor_name} to zenodo')

		if dry_run: std_out(f'Dry run. Verify output before setting dry_run to False', 'WARNING')
		# Sandbox or not
		if sandbox: 
			std_out(f'Using sandbox. Verify output before setting sandbox to False', 'WARNING')
			base_url = self.config['urls']['ZENODO_SANDBOX_BASE_URL']
		else: base_url = self.config['urls']['ZENODO_REAL_BASE_URL']
		
		if '.yaml' not in upload_descriptor_name: upload_descriptor_name = upload_descriptor_name + '.yaml'
		
		with open (join(paths['dataDirectory'], 'uploads', upload_descriptor_name), 'r') as descriptor_file:
			descriptor = yaml.load(descriptor_file)

		self.available_tests()

		for key in descriptor:

			# Set options for processed and raw uploads
			stage_list = ['base']
			if 'options' in descriptor[key].keys(): options = descriptor[key]['options']
			else: options = {'include_processed_data': True, 'include_footer_doi': True}
			if options['include_processed_data']: stage_list.append('processed')
			std_out(f'Options {options}')

			# Fill template
			if 'upload_type' in descriptor[key].keys(): upload_type = descriptor[key]['upload_type']
			else: raise SystemError('Upload type not set')

			metadata = fill_template(descriptor[key], key, upload_type = upload_type)
			
			# Get submission ID
			if not dry_run: submission_id = get_submission_id(metadata, base_url)
			else: submission_id = 0

			if submission_id is not None:
				# Dataset upload
				if upload_type == 'dataset':
					# Get the tests to upload
					tests = descriptor[key]['tests']
					
					# Get url where to post the files
					url = f"{base_url}/api/deposit/depositions/{submission_id}/files"
					
					for test in tests:
						# Get test path
						std_out(f'Uploading data from test {test}')
						test_path = self.availableTests[test]

						# Upload the test descriptor
						with open (join(test_path, 'test_description.yaml'), 'r') as test_descriptor: 
							yaml_test_descriptor = yaml.load(test_descriptor)
						
						upload_metadata = {'name': f'test_description_{test}.yaml'}
						files = {'file': open(join(test_path, 'test_description.yaml'), 'rb')}
						file_size = getsize(join(test_path, 'test_description.yaml'))/(1024*1024.0*1024)
						if file_size > 50: std_out(f'File size for {test} is over 50Gb ({file_size})', 'WARNING')
						if not dry_run: status_code = upload_file(url, upload_metadata, files)
						else: status_code = 200
						
						if status_code > 210: 
							std_out ("Error happened during file upload, status code: " + str(status_code), 'ERROR')
							return
						else:
							std_out(f"{upload_metadata['name']} submitted with submission ID = {submission_id} (DOI: 10.5281/zenodo.{submission_id})" ,"SUCCESS")    
						
						# Load the api devices to have them up to date in the cache
						if any(yaml_test_descriptor['devices'][device]['source'] == 'api' for device in yaml_test_descriptor['devices'].keys()): self.load_test(test, options = {'store_cached_API': True})
						
						for device in yaml_test_descriptor['devices'].keys():
							std_out(f'Uploading data from device {device}')
							
							for file_stage in stage_list:
								file_path = ''
								try:

									# Find evice files
									if file_stage == 'processed' and yaml_test_descriptor['devices'][device]['type'] != 'OTHER': 
										file_name = f'{device}.csv'
										file_path = join(test_path, 'processed', file_name)
										upload_metadata = {'name': f'{device}_PROCESSED.csv'}
									elif file_stage == 'base':
										if 'csv' in yaml_test_descriptor['devices'][device]['source']:
											file_name = yaml_test_descriptor['devices'][device]['fileNameProc']
											file_path = join(test_path, file_name)
										elif yaml_test_descriptor['devices'][device]['source'] == 'api':
											file_name = f'{device}.csv'
											file_path = join(test_path, 'cached', file_name)
										upload_metadata = {'name': file_name}

									if file_path != '':

										files = {'file': open(file_path, 'rb')}
										file_size = getsize(file_path)/(1024*1024.0*1024)
										if file_size > 50: std_out(f'File size for {file_name} over 50Gb ({file_size})', 'WARNING')
										if not dry_run: status_code = upload_file(url, upload_metadata, files)
										else: status_code = 200
										
										if status_code > 210: 
											std_out ("Error happened during file upload, status code: " + str(status_code), 'ERROR')
											return

										std_out(f"{upload_metadata['name']} submitted with submission ID = {submission_id} (DOI: 10.5281/zenodo.{submission_id})" ,"SUCCESS")    
								except:
									if not exists(file_path): std_out(f'File {file_name} does not exist (type = {file_stage}). Skipping', 'ERROR')
									# print_exc()
									pass
								# The submission needs an additional "Publish" step. This can also be done from a script, but to be on the safe side, it is not included. (The attached file cannot be changed after publication.)
					
					# Check if we have a report in the keys
					if 'report' in descriptor[key].keys():
						for file_name in descriptor[key]['report']:
							file_path = join(paths['dataDirectory'], 'uploads', file_name)
							if options['include_footer_doi'] and file_name.endswith('.pdf'):
								output_file_path = file_path[:file_path.index('.pdf')] + '_doi.pdf'
								include_footer(file_path, output_file_path, link = f'https://doi.org/10.5281/zenodo.{submission_id}')
								file_path = output_file_path
							
							upload_metadata = {'name': file_name}
							files = {'file': open(file_path, 'rb')}
							file_size = getsize(file_path)/(1024*1024.0*1024)
							if file_size > 50: std_out(f'File size for {file_name} is over 50Gb({file_size})', 'WARNING')
							if not dry_run: status_code = upload_file(url, upload_metadata, files)
							else: status_code = 200

							if status_code > 210: 
								std_out ("Error happened during file upload, status code: " + str(status_code), 'ERROR')
								return

							std_out(f"{upload_metadata['name']} submitted with submission ID = {submission_id} (DOI: 10.5281/zenodo.{submission_id})" ,"SUCCESS")

				if upload_type == 'publication':
					std_out('Not implemented')
				std_out(f'Submission completed - (DOI: 10.5281/zenodo.{submission_id})', 'SUCCESS')
				std_out(f'------------------------------------------------------------', 'SUCCESS')