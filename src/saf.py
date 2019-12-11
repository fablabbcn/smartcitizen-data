import traceback

import os
from os import pardir, getcwd, makedirs, mkdir, walk
from os.path import join, abspath, normpath, basename, exists, dirname, getsize
from termcolor import colored
from datetime import datetime, timedelta
from tabulate import tabulate
import re
import yaml, json

from src.data.constants import *
import pandas as pd
import numpy as np

import joblib
from src.secrets import *

class saf:

	def __init__(self, verbose = True):
		
		self.rootDirectory = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
		self.dataDirectory = join(self.rootDirectory, 'data')
		self.interimDirectory = join(self.dataDirectory, 'interim')
		self.modelDirectory = join(self.rootDirectory, 'models')
		self.verbose = verbose

		# Load configuration file
		try:
			config_path = join(self.rootDirectory, 'src', 'config.yaml')
			with open(config_path, 'r') as config_yaml:
				self.std_out(f'Loading configuration file from: {config_path}')
				self.config = yaml.load(config_yaml, Loader=yaml.SafeLoader)
				self.std_out('Loaded configuration file', 'SUCCESS')
		except:
			raise SystemError('Problem loading configuration file')

		# Load calibration file
		try:
			with open(join(self.interimDirectory, 'sensorData.yaml'), 'r') as yml:
				self.devices_database = yaml.load(yml)
				self.std_out(f'Loading devices data file from: {self.interimDirectory}')
		except:
			raise SystemError('Problem loading calibration file')

		# Load sensor names from sensors.h
		sensor_names_21 = self.get_sensor_names(self.config['urls']['SENSOR_NAMES_URL_21'], join(self.interimDirectory), 'sensornames_21', self.config['data']['RELOAD_NAMES'])
		sensor_names_20 = self.get_sensor_names(self.config['urls']['SENSOR_NAMES_URL_20'], join(self.interimDirectory), 'sensornames_20', self.config['data']['RELOAD_NAMES'])
		self.current_names = {**sensor_names_21, **sensor_names_20}
	
	def std_out(self, msg, type_message = None, force = False):
		if self.verbose or force: 
			if type_message is None: print(msg)	
			elif type_message == 'SUCCESS': print(colored(msg, 'green'))
			elif type_message == 'WARNING': print(colored(msg, 'yellow')) 
			elif type_message == 'ERROR': print(colored(msg, 'red'))

	def get_sensor_names(self, sensorsh, json_path, file_name, reload_names = True):
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
									sensorID = re.sub(' ','', line_tokenized_sublist[5])
									if len(line_tokenized_sublist)>9:
										sensor_names[sensorID] = dict()
										sensor_names[sensorID]['SensorLocation'] = re.sub(' ', '', line_tokenized_sublist[0])
										sensor_names[sensorID]['shortTitle'] = re.sub(' ', '', line_tokenized_sublist[3])
										sensor_names[sensorID]['title'] = line_tokenized_sublist[4]
										sensor_names[sensorID]['id'] = re.sub(' ', '', line_tokenized_sublist[5])
										sensor_names[sensorID]['unit'] = line_tokenized_sublist[-1]
				# Save everything to the most recent one
				with open(names_dict, 'w') as fp:
					json.dump(sensor_names, fp)
				self.std_out ('Saved updated sensor names and dumped into', names_dict)

			except:
			    # Load sensors
			    # traceback.print_exc()
			    with open(names_dict) as handle:
			        sensor_names = json.loads(handle.read())
			    self.std_out ('No connection - Retrieving local version for sensors names')

		else:

		    with open(names_dict) as handle:
		        sensor_names = json.loads(handle.read())	

		return sensor_names

	def export_CSV_file(self, path, file_name, df, forced_overwrite = False):

		# If path does not exist, create it
		if not exists(path):
			mkdir(path)

		# If file does not exist 
		if not exists(path + '/' + file_name + '.csv') or forced_overwrite:
			df.to_csv(path + '/' + file_name + '.csv', sep=",")
			self.std_out('File saved to: \n' + path + '/' + file_name +  '.csv', 'SUCCESS')
		else:
			self.std_out("File Already exists - delete it first, I was not asked to overwrite anything!", 'WARNING')

	def read_CSV_file(self, file_path, location, frequency, clean_na, clean_na_method, refIndex = ''):
		# Read pandas dataframe
		df = pd.read_csv(file_path, verbose=False, skiprows=[1])

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
		
		# # Resample
		df = df.resample(frequency, limit = 1).mean()

		# Remove na
		if clean_na:
			if clean_na_method == 'fill':
				df = df.fillna(method='ffill')
				self.std_out('Filling NaN')
			elif clean_na_method == 'drop':
				df.dropna(axis = 0, how='all', inplace=True)
				self.std_out('Dropping NaN')

		return df