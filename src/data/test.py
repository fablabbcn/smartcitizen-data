from shutil import copyfile
import io, pytz, time

from dateutil import parser
import csv

from src.data.device import device_wrapper

from src.saf import *


# name = self.date.strftime('%Y'), self.date.strftime('%m'), self.name

class test_wrapper:
	
	def __init__(self, test_name = '' , data = None):

		if data is None: 
			from src.data.data import data_wrapper
			self.data = data_wrapper()
		else: self.data = data

		self.name = test_name
		self.path = join(self.data.rootDirectory, 'data', 'processed', self.name[:4], self.name[5:7], self.name)
		self.devices = dict()
		self.descriptor_file = dict()
		self.cached_info = dict()
		self.ready_to_model = False
		self.options = self.data.config['data']

	def create (self, details, devices):
		# Create folder structure under data subdir
		if not exists(self.path):
			makedirs(self.path)
		else: 
			raise SystemError('Test already exists with this name. Since you want to create a new test, I will stop here so that I avoid to overwrite')
		
		self.data.std_out('Creating new test')
		self.descriptor_file['test'] = dict()

		self.add_details(details)
		for device in devices: self.add_device(device)

		self.descriptor_file['test']['id'] = self.name

		self.process_files()		

		# Create yaml with test description
		with open(join(self.path, 'test_description.yaml'), 'w') as yaml_file:
			yaml.dump(self.descriptor_file, yaml_file)
			
		self.data.std_out ('Test Creation Finished', 'SUCCESS')

	def update(self, details, devices):
		with open(join(self.path, 'test_description.yaml'), 'r') as yml:
			self.descriptor_file = yaml.load(yml, Loader=yaml.BaseLoader)

		self.add_details(details)
		for device in devices: self.add_device(device)

		self.process_files()		

		# Create yaml with test description
		with open(join(self.path, 'test_description.yaml'), 'w') as yaml_file:
			yaml.dump(self.descriptor_file, yaml_file)
			
		self.data.std_out ('Test Update Finished', 'SUCCESS')

	def add_details(self, details):
		'''
			details: a dict containing the information about the project. Minimum of:
				- project
				- commit
				- author
				- test_type
				- report
				- comment
		'''
		if 'test' not in self.descriptor_file.keys(): self.descriptor_file['test'] = dict()
		try:
			for detail in details.keys():
				
				self.descriptor_file['test'][detail] = details[detail]
		except:
			self.data.std_out ('Add details NOK', 'ERROR')
			traceback.print_exc()
			return
		else:
			self.data.std_out ('Add details OK', 'SUCCESS')

	def add_device(self, device):

		if 'devices' not in self.descriptor_file['test'].keys(): self.descriptor_file['test']['devices'] = dict()

		try:
			if device.type == 'KIT' or device.type == 'STATION':
				if 'kits' not in self.descriptor_file['test']['devices'].keys(): self.descriptor_file['test']['devices']['kits'] = dict()
				self.descriptor_file['test']['devices']['kits'][device.name] = dict()
				self.descriptor_file['test']['devices']['kits'][device.name]['name'] = device.name
				self.descriptor_file['test']['devices']['kits'][device.name]['type'] = device.type
				self.descriptor_file['test']['devices']['kits'][device.name]['SCK'] = device.version
				self.descriptor_file['test']['devices']['kits'][device.name]['PM'] = device.pm_sensor
				self.descriptor_file['test']['devices']['kits'][device.name]['location'] = device.location
				self.descriptor_file['test']['devices']['kits'][device.name]['frequency'] = device.frequency

				if device.type == 'STATION': self.descriptor_file['test']['devices']['kits'][device.name]['alphasense'] = device.alphasense

				self.descriptor_file['test']['devices']['kits'][device.name]['source'] = device.source
				if 'csv' in device.source:
					self.descriptor_file['test']['devices']['kits'][device.name]['name'] = device.name
					self.descriptor_file['test']['devices']['kits'][device.name]['fileNameRaw'] = device.raw_data_file
					self.descriptor_file['test']['devices']['kits'][device.name]['fileNameInfo'] = device.info_file
					self.descriptor_file['test']['devices']['kits'][device.name]['fileNameProc'] = self.name + '_' + device.type + '_' + device.name + '.csv'
				elif 'api' in device.source:
					self.descriptor_file['test']['devices']['kits'][device.name]['device_id'] = device.name
			elif device.type == 'ANALYSER':
				if 'reference' not in self.descriptor_file['test']['devices']: self.descriptor_file['test']['devices']['reference'] = dict()
				
				self.descriptor_file['test']['devices']['reference'][device.name] = dict()
				self.descriptor_file['test']['devices']['reference'][device.name]['name'] = device.name
				self.descriptor_file['test']['devices']['reference'][device.name]['fileNameRaw'] = device.raw_data_file
				self.descriptor_file['test']['devices']['reference'][device.name]['fileNameProc'] = self.name + '_' + device.name + '_REF.csv'
				self.descriptor_file['test']['devices']['reference'][device.name]['index'] = device.index
				self.descriptor_file['test']['devices']['reference'][device.name]['channels'] = device.channels
				self.descriptor_file['test']['devices']['reference'][device.name]['location'] = device.location
				self.descriptor_file['test']['devices']['reference'][device.name]['source'] = device.source
				self.descriptor_file['test']['devices']['reference'][device.name]['type'] = device.type
				self.descriptor_file['test']['devices']['reference'][device.name]['equipment'] = device.equipment
		except:
			self.data.std_out (f'Error adding device files for {device.name}', 'ERROR')
			traceback.print_exc()
			return
		else:
			self.data.std_out (f'Added device files for {device.name}', 'SUCCESS')
	
	def process_files(self):
		self.data.std_out('Processing files')
		
		def get_raw_files():
				list_raw_files = []
				if 'kits' in self.descriptor_file['test']['devices'].keys():
					for kit in self.descriptor_file['test']['devices']['kits'].keys():
						if 'csv' in self.descriptor_file['test']['devices']['kits'][kit]['source']:
							list_raw_files.append(self.descriptor_file['test']['devices']['kits'][kit]['fileNameRaw'])
						
				if 'reference' in self.descriptor_file['test']['devices'].keys():
					for reference in self.descriptor_file['test']['devices']['reference'].keys():
						list_raw_files.append(self.descriptor_file['test']['devices']['reference'][reference]['fileNameRaw'])
				
				return list_raw_files    
		
		def copy_raw_files(_raw_src_path, _raw_dst_path, _list_raw_files):
				try: 

					for item in _list_raw_files:
						s = join(_raw_src_path, item)
						d = join(_raw_dst_path, item)
						copyfile(s, d)

					self.data.std_out('Copy raw files: OK', 'SUCCESS')
					
					return True
				
				except:
					self.data.std_out('Problem copying raw files', 'ERROR')
					traceback.print_exc()
					return False
				
		def date_parser(s, a):
			return parser.parse(s).replace(microsecond=int(a[-3:])*1000)

		# Define Paths
		raw_src_path = join(self.data.rootDirectory, 'data', 'raw')
		raw_dst_path = join(self.path, 'RAW_DATA')    
		
		# Create Paths
		if not os.path.exists(raw_dst_path):
			os.makedirs(raw_dst_path)
		
		list_raw_files = get_raw_files()
		# Copy raw files and process data
		if copy_raw_files(raw_src_path, raw_dst_path, list_raw_files):
			# Process references
			if 'reference' in self.descriptor_file['test']['devices']:
				for reference in self.descriptor_file['test']['devices']['reference']:
					self.data.std_out ('Processing reference: {}'.format(reference))
					src_path = join(raw_src_path, self.descriptor_file['test']['devices']['reference'][reference]['fileNameRaw'])
					dst_path = join(self.path, self.name + '_' + str(reference) + '_REF.csv')
					
					# Time Name
					timeName = self.descriptor_file['test']['devices']['reference'][reference]['index']['name']
					
					# Load Dataframe
					df = pd.read_csv(src_path, verbose=False, skiprows=[1]).set_index(timeName)
					df.index = pd.to_datetime(df.index)
					df.sort_index(inplace=True)
					
					df = df.groupby(pd.Grouper(freq = self.descriptor_file['test']['devices']['reference'][reference]['index']['frequency'])).aggregate(np.mean)
					
					# Remove Duplicates and drop unnamed columns
					df = df[~df.index.duplicated(keep='first')]
					df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
					
					# Export to csv in destination path
					df.to_csv(dst_path, sep=",")

			
			# Process kits
			if 'kits' in self.descriptor_file['test']['devices']:
				for kit in self.descriptor_file['test']['devices']['kits']:
					if 'csv' in self.descriptor_file['test']['devices']['kits'][kit]['source']:
						self.data.std_out ('Processing csv from device {}'.format(kit))
						src_path = join(raw_src_path, self.descriptor_file['test']['devices']['kits'][kit]['fileNameRaw'])
						dst_path = join(self.path, self.name + '_' + self.descriptor_file['test']['devices']['kits'][kit]['type'] + '_' + str(kit) + '.csv')
						
						# Read file csv
						if self.descriptor_file['test']['devices']['kits'][kit]['source'] == 'csv_new':
							skiprows_pd = range(1, 4)
							index_name = 'TIME'
							df = pd.read_csv(src_path, verbose=False, skiprows=skiprows_pd, encoding = 'utf-8', sep=',')

						elif self.descriptor_file['test']['devices']['kits'][kit]['source'] == 'csv_old':
							index_name = 'Time'
							df = pd.read_csv(src_path, verbose=False, encoding = 'utf-8')
							
						elif self.descriptor_file['test']['devices']['kits'][kit]['source'] == 'csv_ms':
							index_name = 'Time'
							df = pd.read_csv(src_path, verbose=False, encoding = 'utf-8', parse_dates=[[0,1]], date_parser=date_parser)
						
						# Find name in case of extra weird characters
						for column in df.columns:
							if index_name in column: index_found = column
								
						df.set_index(index_found, inplace = True)
						# df.index = pd.to_datetime(df.index).tz_convert(self.descriptor_file['test']['devices']['kits'][kit]['location'])
						df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(self.descriptor_file['test']['devices']['kits'][kit]['location'])

						df.sort_index(inplace=True)
								
						# Remove Duplicates and drop unnamed columns
						df = df[~df.index.duplicated(keep='first')]
						df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
							
						df.to_csv(dst_path, sep=",")
						
						## Import units and ids
						if self.descriptor_file['test']['devices']['kits'][kit]['source'] == 'csv_new':
							self.data.std_out('Processing units')
							dict_header = dict()
							try:
								with open(src_path, 'rt') as csvfile:
									readercsv = csv.reader(csvfile, delimiter = ',')
									line = 0
								
									header = next(readercsv)[1:]
									unit = next(readercsv)[1:]
									ids = next(readercsv)[1:]
								
									for key in header:
										dict_header[key] = dict()
										dict_header[key]['unit'] = unit[header.index(key)]
										dict_header[key]['id'] = ids[header.index(key)]
									
									self.descriptor_file['test']['devices']['kits'][kit]['metadata'] = dict_header
							except:
								with open(src_path, 'rb') as csvfile:
									readercsv = csv.reader(csvfile, delimiter = ',')
									line = 0
								
									header = next(readercsv)[1:]
									unit = next(readercsv)[1:]
									ids = next(readercsv)[1:]
								
									for key in header:
										dict_header[key] = dict()
										dict_header[key]['unit'] = unit[header.index(key)]
										dict_header[key]['id'] = ids[header.index(key)]
									
									self.descriptor_file['test']['devices']['kits'][kit]['metadata'] = dict_header
							
						## Load txt info
						if self.descriptor_file['test']['devices']['kits'][kit]['fileNameInfo'] != '':
							self.data.std_out('Loading txt info')
							src_path_info = join(raw_src_path, self.descriptor_file['test']['devices']['kits'][kit]['fileNameInfo'])
							dict_info = dict()
							with open(src_path_info, 'rb') as infofile:
								for line in infofile:
									line = line.strip('\r\n')
									splitter = line.find(':')
									dict_info[line[:splitter]]= line[splitter+2:] # Accounting for the space
							   
							self.descriptor_file['test']['devices']['kits'][kit]['info'] = dict_info
						else:
							self.data.std_out('No txt info available', 'WARNING')

			self.data.std_out('Processing files OK', 'SUCCESS')
		self.data.std_out(f'Test {self.name} path: {self.path}')

	def set_options(self, options):
		if 'load_cached_API' in options.keys(): self.options['load_cached_API'] = options['load_cached_API']
		if 'store_cached_API' in options.keys(): self.options['store_cached_API'] = options['store_cached_API']

	def load (self, test_options):

		def process_device_date(date, location):

			if date is not None:
				result_date = pd.to_datetime(date, utc = True)
				if result_date.tzinfo is None: result_date = result_date.tz_convert(location)
			else: 
				result_date = None

			return result_date
		
		self.data.std_out(f'Loading test {self.name}', force = True)
		descriptor_file_path = join(self.path, 'test_description.yaml')

		with open(descriptor_file_path, 'r') as stream:
			self.descriptor_file = yaml.load(stream, Loader=yaml.BaseLoader)

		# Add devices
		for key in self.descriptor_file['test']['devices']['kits'].keys():
			self.devices[key] = device_wrapper(self.descriptor_file['test']['devices']['kits'][key], self.data)

		if 'reference' in self.descriptor_file['test']['devices'].keys():
			for key in self.descriptor_file['test']['devices']['reference'].keys():
				self.devices[key] = device_wrapper(self.descriptor_file['test']['devices']['reference'][key], self.data)

		# Set options
		self.set_options(test_options)
		
		# Check if we need the cached info file for the API
		if self.options['load']['load_cached_API']:
			try:
				with open (join(self.path, 'cached', 'cached_info.json')) as handle:
					self.cached_info = json.loads(handle.read())			
			except:
				self.data.std_out('No cached info file', 'WARNING')
				self.cached_info = dict()
			else:
				self.data.std_out('Loaded cached info file', 'SUCCESS')
		else:
			self.cached_info = dict()
			load_API = True

		for key in self.devices.keys():
			
			device = self.devices[key]
			self.data.std_out(f'Loading device {device.name}')

			min_date_device = process_device_date(device.min_date, device.location)
			max_date_device = process_device_date(device.max_date, device.location)
			device_options = self.options['load']['devices']

			# Name convertion
			target_names = list()
			test_names = list()

			if device.metadata is not None:
				self.data.std_out('Found metadata', 'SUCCESS')
				
				# Use metadata to convert names
				for item_test in device.metadata:
					if device.metadata[item_test]['id'] == 0: continue

					for item_target in self.data.current_names:
						if self.data.current_names[item_target]['id'] == device.metadata[item_test]['id'] and item_test not in test_names:
							target_names.append(self.data.current_names[item_target]['shortTitle'])
							test_names.append(item_test)  
			else:
				self.data.std_out('No metadata found - skipping', 'WARNING')

			# If device comes from API, pre-check dates
			if device.source == 'api':
				
				# Get last reading from API
				last_reading_api = process_device_date(device.api_device.get_date_last_reading(), device.location)

				if self.options['load']['load_cached_API']:

					try:
						device_cached_file = join(self.path, 'cached', device.name  + '.csv')

						location_cached = self.cached_info[device.name]['location']

						if location_cached is None:
							self.data.std_out(f'Requestion device {device.name} locations to API', 'WARNING')
							device.location = device.api_device.get_device_location()
						else:
							device.location = location_cached

						device.load(options = None, path = join(self.path, 'cached'))

					except:

						self.data.std_out(f'No valid cached data. Requesting device {device.name} to API', 'WARNING')
						traceback.print_exc()
						min_date_to_load = min_date_device
						max_date_to_load = max_date_device
						load_API = True

					else:
						
						self.data.std_out(f'Loaded cached files from: {device_cached_file}', 'SUCCESS')

						# Get last reading from cached
						last_reading_cached = process_device_date(device.readings.index[-1], device.location)

						# Check which dates to load
						if max_date_device is not None:
							self.data.std_out(f'Max date in test {max_date_device}')
							# Check what where we need to load data from, if any
							if last_reading_cached < max_date_device and last_reading_api > last_reading_cached + timedelta(hours=1):
								load_API = True
								combine_cache_API = True
								min_date_to_load = last_reading_cached
								max_date_to_load = min(max_date_device, last_reading_api)
								self.data.std_out('Loading new data from API')
							else:
								load_API = False
								self.data.std_out('No need to load new data from API', 'SUCCESS')
						else:
							# If no test data specified, check the last reading in the API
							if last_reading_api > last_reading_cached + timedelta(hours=self.data.config['data']['CACHED_DATA_MARGIN_HOURS']):
								load_API = True
								combine_cache_API = True
								min_date_to_load = last_reading_cached
								max_date_to_load = last_reading_api
								self.data.std_out('Loading new data from API', 'WARNING')
							else:
								load_API = False
								self.data.std_out('No need to load new data from API', 'SUCCESS')
				else:
					min_date_to_load = min_date_device
					max_date_to_load = max_date_device

				if load_API:
					self.data.std_out('Downloading device from API')
					if device.location is None: device.location = device.api_device.get_device_location()

					if last_reading_api is not None:
						# Check which max date to load
						if max_date_to_load is not None:
							if max_date_to_load > last_reading_api:
								# Not possible to load what has not been stored
								max_date_to_load = last_reading_api
							else:
								# Only request what we asked for
								max_date_to_load = max_date_to_load
						else:
							# Just put None and we will handle it later
							max_date_to_load = last_reading_api

						# Check which min date to load
						if min_date_to_load is not None:
							self.data.std_out('First reading requested: {}'.format(min_date_to_load))
							if min_date_to_load < last_reading_api:
								self.data.std_out('Requesting up to max available date in the API {}'.format(last_reading_api))
								min_date_to_load = min_date_to_load
							else:
								self.data.std_out('Discarding device. Min date requested is after last reading', 'WARNING')
								continue
						else:
							self.data.std_out('Requesting all available data', 'WARNING')
							# Just put None and we will handle it later
							min_date_to_load = None

					else:
						self.data.std_out('Discarding device. No max date available', 'WARNING')
						continue

					device_options['min_date'] = min_date_to_load
					device_options['max_date'] = max_date_to_load

					device.load(options = device_options)

			# Rename columns
			elif 'csv' in device.source:
				device.load(device_options, path = self.path)
				
				if len(target_names) == len(test_names) and len(target_names) > 0:
					for i in range(len(target_names)):
						if not (test_names[i] == '') and not (test_names[i] == target_names[i]) and test_names[i] in device.readings.columns:
							device.readings.rename(columns={test_names[i]: target_names[i]}, inplace=True)
							self.data.std_out('Renaming column {} to {}'.format(test_names[i], target_names[i]))

			if self.options['load']['store_cached_API'] and device.loaded_OK and device.source == 'api' and load_API:
				self.data.std_out(f'Caching files for {device.name}')

				if device.name not in self.cached_info.keys(): self.cached_info[device.name] = dict()
				self.cached_info[device.name]['location'] = device.location
				if device.type == 'STATION': self.cached_info[device.name]['alphasense'] = device.alphasense

				cached_file_path = join(self.path, 'cached')
				if not exists(cached_file_path):
					self.data.std_out('Creating path for exporting cached data')
					makedirs(cached_file_path)

				# Dump what we processed so far, the fully processed CSV will be saved later on
				with open(join(cached_file_path, 'cached_info.json'), 'w') as file:
					json.dump(self.cached_info, file)
					
				self.data.export_CSV_file(cached_file_path, device.name, device.readings, forced_overwrite = True)

			self.data.std_out(f'Device {device.name} has been loaded', 'SUCCESS')




