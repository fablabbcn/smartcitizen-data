from src.saf import std_out, dict_fmerge, read_csv_file
from src.saf import BLUEPRINTS
from os.path import join

import pandas as pd
from traceback import print_exc

class device_wrapper:
	def __init__(self, blueprint, descriptor):
		self.blueprint = blueprint

		# Set attributes
		for bpitem in BLUEPRINTS[blueprint]: self.__setattr__(bpitem, BLUEPRINTS[blueprint][bpitem]) 
		for ditem in descriptor.keys():
			if type(self.__getattribute__(ditem)) == dict: self.__setattr__(ditem, dict_fmerge(self.__getattribute__(ditem), descriptor[ditem]))
			else: self.__setattr__(ditem, descriptor[ditem])

		# Add API handler if needed
		if self.source == 'api':
			hmod = __import__('src.data.api', fromlist=['data.api'])
			Hclass = getattr(hmod, self.sources[self.source]['handler'])
			# Create object
			self.api_device = Hclass(self.id)

		self.readings = pd.DataFrame()
		self.loaded = False
		self.options = dict()

	def check_overrides(self, options = {}):
		if 'min_date' in options.keys(): self.options['min_date'] = options['min_date'] 
		else: self.options['min_date'] = self.min_date
		if 'max_date' in options.keys(): self.options['max_date'] = options['max_date']
		else: self.options['max_date'] = self.max_date
		if 'clean_na' in options.keys(): self.options['clean_na'] = options['clean_na']
		else: self.options['clean_na'] = self.clean_na
		if 'frequency' in options.keys(): self.options['frequency'] = options['frequency']
		else: self.options['frequency'] = self.frequency		
   
	def load(self, options = None, path = None):
		# Add test overrides if we have them, otherwise set device defaults
		if options is not None: self.check_overrides(options)
		else: self.check_overrides()

		try:
			if self.source == 'csv':
				self.readings = self.readings.combine_first(read_csv_file(join(path, self.processed_data_file), self.location, self.options['frequency'], 
															self.options['clean_na'], self.sources[self.source]['index']))

			elif 'api' in self.source:
				if path is None:
					df = self.api_device.get_device_data(self.options['min_date'], self.options['max_date'], self.options['frequency'], self.options['clean_na'])
					# API Device is not aware of other csv index data, so make it here 
					df = df.reindex(df.index.rename(self.sources['csv']['index']))
					self.readings = self.readings.combine_first(df)
				else:
					# Cached case
					self.readings = self.readings.combine_first(read_csv_file(join(path, self.id + '.csv'), self.location, self.options['frequency'], 
															self.options['clean_na'], self.sources['csv']['index']))

			# # Convert units
			# if self.type == 'OTHER': self.convert_units(append_to_name = 'CONV')
		
		except FileNotFoundError:
			std_out('Cached file does not exist', 'ERROR')
			self.loaded = False
		except:
			print_exc()
			self.loaded = False
		else:
			if self.readings is not None: self.loaded = True

		return self.loaded

	# TODO
	def convert_names(self):
		print ('Not yet')
	
	# TODO
	# def convert_units(self, append_to_name = ''):
	# 	logging.info('Checking if units need to be converted')

	# 	for channel_nmbr in range(len(self.channels['target_channel_names'])):
	# 		target_channel_name = self.channels['target_channel_names'][channel_nmbr]
	# 		source_channel_name = self.channels['source_channel_names'][channel_nmbr]
	# 		unit = self.channels['units'][channel_nmbr]
	# 		target_unit = None

	# 		for channel_convert in channel_LUT: 
	# 			if channel_convert[0] == target_channel_name: 
	# 				molecular_weight = channel_convert[1]
	# 				target_unit = channel_convert[2]

	# 		# Get convertion factor
	# 		if target_unit is not None:
	# 			if unit == target_unit:
	# 				convertion_factor = 1
	# 				logging.info('No unit convertion needed for {}'.format(target_channel_name))
	# 			else:
	# 				for item_to_convert in convertion_LUT:
	# 					if item_to_convert[0] == unit and item_to_convert[1] == target_unit:
	# 						convertion_factor = item_to_convert[2]/molecular_weight
	# 					elif item_to_convert[1] == unit and item_to_convert[0] == target_unit:
	# 						convertion_factor = 1.0/(item_to_convert[2]/molecular_weight)
				
	# 				logging.info('Converting {} from {} to {}'.format(target_channel_name, unit, target_unit))
	# 			self.readings.loc[:, target_channel_name + '_' + append_to_name] = self.readings.loc[:, source_channel_name]*convertion_factor
	# 		else:
	# 			logging.warning('No unit convertion needed for {}. Actual channel name is not in look-up tables'.format(target_channel_name))
		
	def capture(self):
		logging.error('Not yet')