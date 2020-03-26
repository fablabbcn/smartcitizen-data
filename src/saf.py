### ---------------------------------------
### ---------------LOGGING-----------------
### ---------------------------------------

import logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
    ]
)

verbose = True
def std_out(msg, type_message = None, force = False):
	if verbose or force: 
		if type_message is None: 
			print(msg)	
		elif type_message == 'SUCCESS': 
			print(colored(msg, 'green'))
			logging.info(msg)
		elif type_message == 'WARNING': 
			print(colored(msg, 'yellow'))
			logging.warning(msg, 'WARNING')
		elif type_message == 'ERROR': 
			print(colored(msg, 'red'))
			logging.error(msg)

### ---------------------------------------
### ---------------IMPORTS-----------------
### ---------------------------------------

from traceback import print_exc
from termcolor import colored

import re

from os import pardir, makedirs
from os.path import join, abspath, dirname, exists

import pandas as pd
import yaml 
import json

from urllib.request import urlopen
import requests

# INTERNAL
from src.secrets import *

# ### ALL
# from os import pardir, getcwd, makedirs, mkdir, walk
# from os.path import join, abspath, normpath, basename, , dirname, getsize

# from datetime import datetime, timedelta
# from tabulate import tabulate
# import re
# from shutil import copyfile
# import io, pytz, time
# from dateutil import parser
# from urllib.request import urlopen
# import requests
# from tzwhere import tzwhere

# # FILES
# import yaml, json
# import joblib
# import csv

# # INTERNAL
# from src.secrets import *

### ---------------------------------------
### ----------------ZENODO-----------------
### ---------------------------------------

# Urls
ZENODO_SANDBOX_BASE_URL='http://sandbox.zenodo.org'
ZENODO_REAL_BASE_URL='https://zenodo.org'

### ---------------------------------------
### -------------SMART CITIZEN-------------
### ---------------------------------------

# Urls
SENSOR_NAMES_URL_21='https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-21/master/lib/Sensors/Sensors.h'
SENSOR_NAMES_URL_20='https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/master/lib/Sensors/Sensors.h'

# Convertion table from API SC to Pandas
# https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
# https://developer.smartcitizen.me/#get-historical-readings
FREQ_CONV_LUT = (['y','A'],
					    ['M','M'],
					    ['w','W'],
					    ['d','D'],
					    ['h','H'],
					    ['m','Min'],
					    ['s','S'],
					    ['ms','ms'])

# Alphasense ID table (Slot, Working, Auxiliary)
as_ids_table = ([1,'64','65'], 
			 [2,'61','62'], 
			 [3,'67','68'])

# External temperature table (this table is by priority)
th_ids_table = (['EXT_DALLAS','96',''], 
				 ['EXT_SHT31','79', '80'], 
				 ['SENSOR_TEMPERATURE','55','56'],
				 ['GASESBOARD_TEMPERATURE','79', '80'])

# This look-up table is comprised of channels you want always want to have with the same units and that might come from different sources
# i.e. pollutant data in various units (ppm or ug/m3) from different analysers
# The table should be used as follows:
# (['int_channel', 'molecular_weight', 'target_unit'], ...)
# - int_channel is the internal channel that will lately be used in the analysis. This has to be defined in the target_channel_names when creating the test (see below)
# - molecular_weight is only for chemical components. It can be left to 1 for other types of signals
# - target_unit is the unit you want this channel to be and that will be converted in case of it being found in the channels list of your source

channel_LUT = (['CO', 28, 'ppm'],
				['NO', 30, 'ppb'],
				['NO2', 46, 'ppb'],
				['O3', 48, 'ppb'])

# Target channel name definition
# This dict has to be specified when you create a test.
# 'channels': {'source_channel_names' : ('air_temperature_celsius', 'battery_percent', 'calibrated_soil_moisture_percent', 'fertilizer_level', 'light', 'soil_moisture_percent', 'water_tank_level_percent'), 
#              'units' : ('degC', '%', '%', '-', 'lux', '%', '%'),
#              'target_channel_names' : ('TEMP', 'BATT', 'Cal Soil Moisture', 'Fertilizer', 'Soil Moisture', 'Water Level')
# source_channel_names: is the actual name you can find in your csv file
# units: the units of this channel. These will be converted using the LUT below
# target_channel_names: how you want to name your channels after the convertion. A suffix ('_CONV') will be added to them in case they are matching the source csv names

# Can be targetted to convert the units with the channel_LUT below
# This table is used to convert units
# ['from_unit', 'to_unit', 'multiplicative_factor']
# - 'from_unit'/'to_unit' = 'multiplicative_factor'
# It accepts reverse operations - you don't need to put them twice but in reverse
UNIT_CONVERTION_LUT = (['ppm', 'ppb', 1000],
					['mg/m3', 'ug/m3', 1000],
					['mgm3', 'ugm3', 1000],
					['mg/m3', 'ppm', 24.45],
					['mgm3', 'ppm', 24.45],
					['ug/m3', 'ppb', 24.45],
					['ugm3', 'ppb', 24.45],
					['mg/m3', 'ppb', 1000*24.45],
					['mgm3', 'ppb', 1000*24.45],
					['ug/m3', 'ppm', 1./1000*24.45],
					['ugm3', 'ppm', 1./1000*24.45])

### --------------------------------------
### -------------SENSORS DATA-------------
### --------------------------------------
# Units Look Up Table - ['Pollutant', unit factor from ppm to target 1, unit factor from ppm to target 2]
alpha_factors_LUT = (['CO', 1, 0],
						['NO2', 1000, 0],
						['O3', 1000, 1000])

# AlphaDelta PCB factor (converstion from mV to nA)
factor_alphadelta_pcb = 6.36

# Background Concentration (model assumption)
# (from Modelling atmospheric composition in urban street canyons - Vivien Bright, William Bloss and Xiaoming Cai)
background_conc_CO = 0 # ppm
background_conc_NO2 = 8 # ppb
background_conc_OX = 40 # ppb

# Filter Smoothing 
filter_exponential_smoothing = 0.2

### ---------------------------------------
### ----------------PATHS------------------
### ---------------------------------------
paths = dict()
paths['rootDirectory'] = abspath(abspath(join(dirname(__file__), pardir)))
paths['dataDirectory'] = join(paths['rootDirectory'], 'data')
paths['interimDirectory'] = join(paths['dataDirectory'], 'interim')
paths['modelDirectory'] = join(paths['rootDirectory'], 'models')

try:
	paths['toolsDirectory'] = tools_path
	paths['inventoryDirectory'] = inventory_path
except:
	std_out('Cannot use tools and inventory without path in secrets', 'WARNING')
	pass

### ---------------------------------------
### ----------------CONFIG-----------------
### ---------------------------------------
try:
	config_path = join(paths['rootDirectory'], 'src', 'config.yaml')
	with open(config_path, 'r') as config_yaml:
		std_out(f'Loading configuration file from: {config_path}')
		configuration = yaml.load(config_yaml, Loader=yaml.SafeLoader)
		std_out('Loaded configuration file')
except:
	std_out('Error loading configuration file', 'ERROR')
	raise SystemError('Problem loading configuration file')


### ---------------------------------------
### --------------BLUEPRINTS---------------
### ---------------------------------------
try:
	blueprints_path = join(paths['interimDirectory'], 'blueprints.yaml')
	with open(blueprints_path, 'r') as blueprints_yaml:
		std_out(f'Loading blueprints file from: {blueprints_path}')
		BLUEPRINTS = yaml.load(blueprints_yaml, Loader=yaml.SafeLoader)
		std_out('Loaded blueprints file')
except:
	std_out('Error loading blueprints file', 'ERROR')
	raise SystemError('Problem loading blueprints file')

def dict_fmerge(base_dct, merge_dct, add_keys=True):
	"""
	From: https://gist.github.com/CMeza99/5eae3af0776bef32f945f34428669437
	Recursive dict merge.
	Args:
		base_dct (dict) onto which the merge is executed
		merge_dct (dict): base_dct merged into base_dct
		add_keys (bool): whether to add new keys
	Returns:
		dict: updated dict
	"""
	rtn_dct = base_dct.copy()
	if add_keys is False:
		merge_dct = {key: merge_dct[key] for key in set(rtn_dct).intersection(set(merge_dct))}

	rtn_dct.update({
		key: dict_fmerge(rtn_dct[key], merge_dct[key], add_keys=add_keys)
		if isinstance(rtn_dct.get(key), dict) and isinstance(merge_dct[key], dict)
		else merge_dct[key]
		for key in merge_dct.keys()
	})
	return rtn_dct

for blueprint in BLUEPRINTS.keys():
	if 'expands' in BLUEPRINTS[blueprint]: 
		BLUEPRINTS[blueprint] = dict_fmerge(BLUEPRINTS[blueprint], BLUEPRINTS[BLUEPRINTS[blueprint]['expands']])
		BLUEPRINTS[blueprint].pop('expands')
std_out(f'Merged blueprints', 'SUCCESS')

### ---------------------------------------
### -------------SENSOR NAMES--------------
### ---------------------------------------
def get_firmware_names(sensorsh, json_path, file_name, reload_names = True):
	# Directory
	names_dict = join(json_path, file_name + '.json')
	
	if reload_names:
		try:
			# Read only 20000 chars
			data = urlopen(sensorsh).read(20000).decode('utf-8')
			# split it into lines
			data = data.split('\n')
			sensor_names = dict()
			line_sensors = len(data)
			for line in data:
				if 'class AllSensors' in line:
					line_sensors = data.index(line)
					
				if data.index(line) > line_sensors:

					if 'OneSensor' in line and '{' in line and '}' in line and '/*' not in line:
						# Split commas
						line_tokenized =  line.strip('').split(',')

						# Elimminate unnecessary elements
						line_tokenized_sublist = list()
						for item in line_tokenized:
								item = re.sub('\t', '', item)
								item = re.sub('OneSensor', '', item)
								item = re.sub('{', '', item)
								item = re.sub('}', '', item)
								#item = re.sub(' ', '', item)
								item = re.sub('"', '', item)
								
								if item != '' and item != ' ':
									while item[0] == ' ' and len(item)>0: item = item[1:]
								line_tokenized_sublist.append(item)
						line_tokenized_sublist = line_tokenized_sublist[:-1]

						if len(line_tokenized_sublist) > 2:
								shortTitle = re.sub(' ', '', line_tokenized_sublist[3])
								if len(line_tokenized_sublist)>9:
									sensor_names[shortTitle] = dict()
									sensor_names[shortTitle]['SensorLocation'] = re.sub(' ', '', line_tokenized_sublist[0])
									sensor_names[shortTitle]['id'] = re.sub(' ','', line_tokenized_sublist[5])
									sensor_names[shortTitle]['title'] = line_tokenized_sublist[4]
									sensor_names[shortTitle]['unit'] = line_tokenized_sublist[-1]
			# Save everything to the most recent one
			with open(names_dict, 'w') as fp:
				json.dump(sensor_names, fp)
			std_out('Saved updated sensor names and dumped into ' + names_dict, 'SUCCESS')

		except:
			# Load sensors
			print_exc()
			with open(names_dict) as handle:
				sensor_names = json.loads(handle.read())
			std_out('Error. Retrieving local version for sensors names', 'WARNING')

	else:
		std_out('Retrieving local version for sensors names')
		with open(names_dict) as handle:
			sensor_names = json.loads(handle.read())
		if sensor_names is not None: std_out('Local version of sensor names loaded', 'SUCCESS')

	return sensor_names

sensor_names_21 = get_firmware_names(SENSOR_NAMES_URL_21, join(paths['interimDirectory']), 'sensornames_21', configuration['data']['reload_firmware_names'])
sensor_names_20 = get_firmware_names(SENSOR_NAMES_URL_20, join(paths['interimDirectory']), 'sensornames_20', configuration['data']['reload_firmware_names'])
CURRENT_NAMES = {**sensor_names_21, **sensor_names_20}

# Update blueprints
for blueprint in BLUEPRINTS:
	# Skip non sc sensors
    if 'sc' not in blueprint[0:3]: continue
    if 'sensors' in BLUEPRINTS[blueprint]: 
        for sensor in BLUEPRINTS[blueprint]['sensors'].keys():
            BLUEPRINTS[blueprint]['sensors'][sensor]['id'] = CURRENT_NAMES[sensor]['id']
# Update file
try:
	blueprints_path = join(paths['interimDirectory'], 'blueprints.yaml')
	with open(blueprints_path, 'w') as blueprints_yaml:
		std_out(f'Updating blueprints file from: {blueprints_path}')
		yaml.dump(BLUEPRINTS, blueprints_yaml)
		std_out('Updated blueprints file', 'SUCCESS')
except:
	std_out('Error loading blueprints file', 'ERROR')
	raise SystemError('Problem saving blueprints file')

### ---------------------------------------
### -----------CSV LOAD/EXPORT-------------
### ---------------------------------------
def read_csv_file(file_path, location, frequency, clean_na = None, index_name = '', skiprows = None, sep = ',', encoding = 'utf-8'):
	'''
		Reads a csv file and adds cleaning, localisation and resampling
	'''

	# Read pandas dataframe
	df = pd.read_csv(file_path, verbose = False, skiprows = skiprows, sep = ',', encoding = 'utf-8')

	for column in df.columns:
		if index_name in column: 
			df = df.set_index(column)
			break
	# if index in df.columns: df = df.set_index(index)
	# if 'Time' in df.columns:
	# 	df = df.set_index('Time')
	# elif 'TIME' in df.columns:
	# 	df = df.set_index('TIME')
	# elif refIndex != '':
	# 	df = df.set_index(refIndex)

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
	
	# # Resample
	df = df.resample(frequency, limit = 1).mean()

	# Remove na
	if clean_na is not None:
		if clean_na == 'fill':
			df = df.fillna(method='ffill')
			std_out('Filling NaN')
		elif clean_na == 'drop':
			df.dropna(axis = 0, how='all', inplace=True)
			std_out('Dropping NaN')

	return df

def export_csv_file(path, file_name, df, forced_overwrite = False):
	'''
		Exports pandas dataframe to a csv file
	'''

	# If path does not exist, create it
	if not exists(path):
		makedirs(path)

	# If file does not exist 
	if not exists(path + '/' + file_name + '.csv') or forced_overwrite:
		df.to_csv(path + '/' + file_name + '.csv', sep=",")
		std_out('File saved to: \n' + path + '/' + file_name +  '.csv', 'SUCCESS')
	else:
		std_out("File Already exists - delete it first, I was not asked to overwrite anything!", 'WARNING')

	return True
def get_localised_date(date, location):

	if date is not None:
		result_date = pd.to_datetime(date, utc = True)
		if result_date.tzinfo is None: result_date = result_date.tz_convert(location)
	else: 
		result_date = None

	return result_date