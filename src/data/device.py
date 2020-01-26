from src.data.api import api_device
from src.saf import *

class device_wrapper:

	def __init__(self, device_descriptor, data = None):
		
		if data is None: 
			from src.data.data import data_wrapper
			self.data = data_wrapper()
		else: self.data = data
		
		# Type = KIT, STATION or OTHER
		self.type = device_descriptor['type']
		# Source (csv_new, csv_old, csv_ms or api)
		self.source = device_descriptor['source']
		# Location
		if 'location' in device_descriptor.keys(): self.location = device_descriptor['location']
		else: self.location = None
		# Dates
		if 'min_date' in device_descriptor.keys(): self.min_date = device_descriptor['min_date']
		else: self.min_date = None
		if 'max_date' in device_descriptor.keys(): self.max_date = device_descriptor['max_date']
		else: self.max_date = None
		# Metadata
		if 'metadata' in device_descriptor.keys(): self.metadata = device_descriptor['metadata']
		else: self.metadata = None
		# Frequency in pandas format
		if 'index' in device_descriptor.keys(): self.frequency = device_descriptor['index']['frequency']
		else: 
			try: self.frequency = device_descriptor['frequency']
			except: self.frequency = None
		
		# Add properties depending on source
		if 'csv' in self.source:
			self.name = device_descriptor['name']
			self.raw_data_file = device_descriptor['fileNameRaw']
			if 'fileNameInfo' in device_descriptor.keys(): self.info_file = device_descriptor['fileNameInfo']
			try:
				self.processed_file = device_descriptor['fileNameProc']
			except:
				pass
		elif 'api' in self.source:
			self.name = device_descriptor['device_id']
			self.api_device = api_device(device_id = self.name, from_device = True, verbose = self.data.verbose)
		elif 'serial' in self.source:
			self.data.std_out('Not supported yet')

		# Kit or Station properties
		if self.type == 'KIT' or self.type == 'STATION':
			self.version = device_descriptor['version']
			self.pm_sensor = device_descriptor['pm_sensor']
			self.index = {'name': ''}
			if self.type == 'STATION':
				if 'alphasense' in device_descriptor.keys(): self.alphasense = device_descriptor['alphasense']
				elif 'device_history' in device_descriptor.keys(): self.alphasense = self.data.devices_database[device_descriptor['device_history']]['gas_pro_board']
				else: self.alphasense = None; self.data.std_out(f'No alphasense specified in files for {self.name}', 'WARNING')
		# Other devices properties
		elif self.type == 'OTHER':
			self.equipment = device_descriptor['equipment']
			self.channels = device_descriptor['channels']
			self.index = device_descriptor['index']

		self.readings = pd.DataFrame()
		self.loaded_OK = False
		self.options = self.data.config['data']

	def set_options(self, options):
		if 'min_date' in options.keys(): self.options['min_date'] = options['min_date'] 
		else: self.options['min_date'] = None
		if 'max_date' in options.keys(): self.options['max_date'] = options['max_date']
		else: self.options['max_date'] = None
		if 'clean_na' in options.keys(): self.options['clean_na'] = options['clean_na']
		if 'clean_na_method' in options.keys(): self.options['clean_na_method'] = options['clean_na_method']

	def load(self, options, path = None):
		if options is not None: self.set_options(options)
		try:
			if 'csv' in self.source:
				self.readings = self.readings.combine_first(self.data.read_CSV_file(join(path, self.processed_file), self.location, self.options['frequency'], 
															self.options['clean_na'], self.options['clean_na_method'], self.index['name']))

			elif 'api' in self.source:
				if path is None:
					self.readings = self.readings.combine_first(self.api_device.get_device_data(self.options['min_date'], self.options['max_date'], self.options['frequency'], 
															self.options['clean_na'], self.options['clean_na_method'], self.data.current_names))
				else:
					# Normally cached
					self.readings = self.readings.combine_first(self.data.read_CSV_file(join(path, self.name + '.csv'), self.location, self.options['frequency'], 
															self.options['clean_na'], self.options['clean_na_method']))

			# Convert units
			if self.type == 'OTHER': self.convert_units(append_to_name = 'CONV')
		
		except:
			traceback.print_exc()
			pass
		else:
			if self.readings is not None: self.loaded_OK = True

	def convert_units(self, append_to_name = ''):
		self.data.std_out('Checking if units need to be converted')

		for channel_nmbr in range(len(self.channels['target_channel_names'])):
			target_channel_name = self.channels['target_channel_names'][channel_nmbr]
			source_channel_name = self.channels['source_channel_names'][channel_nmbr]
			unit = self.channels['units'][channel_nmbr]
			target_unit = None

			for channel_convert in channel_LUT: 
				if channel_convert[0] == target_channel_name: 
					molecular_weight = channel_convert[1]
					target_unit = channel_convert[2]

			# Get convertion factor
			if target_unit is not None:
				if unit == target_unit:
					convertion_factor = 1
					self.data.std_out('No unit convertion needed for {}'.format(target_channel_name), 'SUCCESS')
				else:
					for item_to_convert in convertion_LUT:
						if item_to_convert[0] == unit and item_to_convert[1] == target_unit:
							convertion_factor = item_to_convert[2]/molecular_weight
						elif item_to_convert[1] == unit and item_to_convert[0] == target_unit:
							convertion_factor = 1.0/(item_to_convert[2]/molecular_weight)
				
					self.data.std_out('Converting {} from {} to {}'.format(target_channel_name, unit, target_unit))
				self.readings.loc[:, target_channel_name + '_' + append_to_name] = self.readings.loc[:, source_channel_name]*convertion_factor
			else:
				self.data.std_out('No unit convertion needed for {}. Actual channel name is not in look-up tables'.format(target_channel_name), 'WARNING')

	def capture(self):
		self.data.std_out('Not yet')

