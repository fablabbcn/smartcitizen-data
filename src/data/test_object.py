from shutil import copyfile
import io, pytz, os, time, datetime
from os.path import dirname, join, abspath
from os import getcwd, pardir
import yaml
import pandas as pd
import numpy as np
import markdown
from dateutil import parser

from IPython.display import display, Markdown, FileLink, FileLinks, clear_output, HTML
from IPython.core.display import HTML
from IPython.display import display, clear_output
from plotly.widgets import GraphWidget
	
import ipywidgets as widgets
from ipywidgets import interact
import csv
from src.data.api_utils import *

class test_object:
	
	def __init__(self, ID, sensorsData):
		self.ID = ID
		self.yaml = {}
		self.yaml['test'] = dict()
		self.yaml['test']['id'] = ID
		self.yaml['test']['devices'] = dict()
		self.yaml['test']['devices']['kits'] = dict()
		self.sensorsData = sensorsData  
	
	def add_details(self, project = 'smartcitizen', commit = '', author = '', type_test = '', report = '', comment = ''):
		try:
			self.yaml['test']['project'] = project
			self.yaml['test']['commit'] = commit
			self.yaml['test']['author'] = author
			self.yaml['test']['type_test'] = type_test
			self.yaml['test']['report'] = report
			self.yaml['test']['comment'] = markdown.markdown(comment)
			print ('Add details OK')

		except:
			print ('Add details NOK')
			pass

	def add_device(self, device, device_type = 'KIT', sck_version = '2.0', pm_sensor = '', alphasense = {}, device_history = None, location = 'Europe/Madrid', device_files = {}):
		try:
			self.yaml['test']['devices']['kits'][device] = dict()
			self.yaml['test']['devices']['kits'][device]['type'] = device_type
			self.yaml['test']['devices']['kits'][device]['SCK'] = sck_version
			self.yaml['test']['devices']['kits'][device]['PM'] = pm_sensor
			self.yaml['test']['devices']['kits'][device]['location'] = location
			
			#### Alphasense
			if alphasense != {}:
				self.yaml['test']['devices']['kits'][device]['alphasense'] = alphasense
			elif device_history != None:
				self.yaml['test']['devices']['kits'][device]['alphasense'] = self.sensorsData[device_history]['gas_pro_board']
			print ('Add device {} OK'.format(device))
			
			source = device_files['source']
			self.yaml['test']['devices']['kits'][device]['source'] = source
			# print (device_files)
			if 'csv' in source:
				self.yaml['test']['devices']['kits'][device]['fileNameRaw'] = device_files['fileNameRaw']
				self.yaml['test']['devices']['kits'][device]['fileNameInfo'] = device_files['fileNameInfo']
				fileNameProc = (self.yaml['test']['id'] + '_' + self.yaml['test']['devices']['kits'][device]['type'] + '_' + str(device) + '.csv')
				self.yaml['test']['devices']['kits'][device]['fileNameProc'] = fileNameProc
				self.yaml['test']['devices']['kits'][device]['frequency'] = device_files['frequency']
				

			elif 'api' in source:
				self.yaml['test']['devices']['kits'][device]['device_id'] = device_files['device_id']
				self.yaml['test']['devices']['kits'][device]['frequency'] = device_files['frequency']
				self.yaml['test']['devices']['kits'][device]['min_date'] = device_files['min_date']
				self.yaml['test']['devices']['kits'][device]['max_date'] = device_files['max_date']

			print ('Add device files {} OK'.format(device))
			
		except:
			print ('Add device files {} NOK'.format(device))
			pass
	
	def add_reference(self, reference, fileNameRaw = '', index = {}, channels = {}, location = ''):
		print ('Adding reference: {}'.format(reference))
		if 'reference' not in self.yaml['test']['devices']:
			self.yaml['test']['devices']['reference'] = dict()
		
		self.yaml['test']['devices']['reference'][reference] = dict()
		self.yaml['test']['devices']['reference'][reference]['fileNameRaw'] = fileNameRaw
		self.yaml['test']['devices']['reference'][reference]['fileNameProc'] = self.yaml['test']['id'] + '_' + str(reference) + '_REF.csv'
		self.yaml['test']['devices']['reference'][reference]['index'] = index
		self.yaml['test']['devices']['reference'][reference]['channels'] = channels
		self.yaml['test']['devices']['reference'][reference]['location'] = location
	
	def process_files(self, _rootDirectory, _newpath):
		
		def get_raw_files():
				list_raw_files = []
				
				if 'kits' in self.yaml['test']['devices']:
					for kit in self.yaml['test']['devices']['kits']:
						if 'csv' in self.yaml['test']['devices']['kits'][kit]['source']:
							list_raw_files.append(self.yaml['test']['devices']['kits'][kit]['fileNameRaw'])
						
				if 'references' in self.yaml['test']['devices']:
					for reference in self.yaml['test']['devices']['reference']:
						list_raw_files.append(self.yaml['test']['devices']['references'][reference]['fileNameRaw'])
						
				return list_raw_files    
		
		def copy_raw_files(_raw_src_path, _raw_dst_path, _list_raw_files):
			
				try: 

					for item in _list_raw_files:
						s = join(_raw_src_path, item)
						d = join(_raw_dst_path, item)
						copyfile(s, d)
					
					return True
				
				except:

					return False
				
		def date_parser(s, a):
			return parser.parse(s).replace(microsecond=int(a[-3:])*1000)
	
		# Define Paths
		raw_src_path = join(_rootDirectory, 'data', 'raw')
		raw_dst_path = join(_newpath, 'RAW_DATA')    
		
		# Create Paths
		if not os.path.exists(raw_dst_path):
			os.makedirs(raw_dst_path)
		
		list_raw_files = get_raw_files()
		# Copy raw files and process data
		if copy_raw_files(raw_src_path, raw_dst_path, list_raw_files):
			# Process references
			if 'reference' in self.yaml['test']['devices']:
				for reference in self.yaml['test']['devices']['reference']:
					print ('Processing reference: {}'.format(reference))
					src_path = join(raw_src_path, self.yaml['test']['devices']['reference'][reference]['fileNameRaw'])
					dst_path = join(_newpath, self.yaml['test']['id'] + '_' + str(reference) + '_REF.csv')
					
					# Time Name
					timeName = self.yaml['test']['devices']['reference'][reference]['index']['name']
					
					# Load Dataframe
					df = pd.read_csv(src_path, verbose=False, skiprows=[1]).set_index(timeName)
					df.index = pd.to_datetime(df.index)
					df.sort_index(inplace=True)
					
					df = df.groupby(pd.Grouper(freq = self.yaml['test']['devices']['reference'][reference]['index']['frequency'])).aggregate(np.mean)
					
					# Remove Duplicates and drop unnamed columns
					df = df[~df.index.duplicated(keep='first')]
					df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
					
					# Export to csv in destination path
					df.to_csv(dst_path, sep=",")
					
			
			# Process kits
			if 'kits' in self.yaml['test']['devices']:
				for kit in self.yaml['test']['devices']['kits']:
					if 'csv' in self.yaml['test']['devices']['kits'][kit]['source']:
						print ('Processing device: {}'.format(kit))
						src_path = join(raw_src_path, self.yaml['test']['devices']['kits'][kit]['fileNameRaw'])
						dst_path = join(_newpath, self.yaml['test']['id'] + '_' + self.yaml['test']['devices']['kits'][kit]['type'] + '_' + str(kit) + '.csv')
						
						# Read file csv
						if self.yaml['test']['devices']['kits'][kit]['source'] == 'csv_new':
							skiprows_pd = range(1, 4)
							index_name = 'TIME'
							df = pd.read_csv(src_path, verbose=False, skiprows=skiprows_pd, encoding = 'utf-8', sep=',')

						elif self.yaml['test']['devices']['kits'][kit]['source'] == 'csv_old':
							index_name = 'Time'
							df = pd.read_csv(src_path, verbose=False, encoding = 'utf-8')
							
						elif self.yaml['test']['devices']['kits'][kit]['source'] == 'csv_ms':
							index_name = 'Time'
							df = pd.read_csv(src_path, verbose=False, encoding = 'utf-8', parse_dates=[[0,1]], date_parser=date_parser)
						
						# Find name in case of extra weird characters
						for column in df.columns:
							if index_name in column: index_found = column
								
						df.set_index(index_found, inplace = True)
						df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(self.yaml['test']['devices']['kits'][kit]['location'])
						df.sort_index(inplace=True)
								
						# Remove Duplicates and drop unnamed columns
						df = df[~df.index.duplicated(keep='first')]
						df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
							
						df.to_csv(dst_path, sep=",")
						
						## Import units and ids
						if self.yaml['test']['devices']['kits'][kit]['source'] == 'csv_new':
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
									
									self.yaml['test']['devices']['kits'][kit]['metadata'] = dict_header
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
									
									self.yaml['test']['devices']['kits'][kit]['metadata'] = dict_header
							
						## Load txt info
						if self.yaml['test']['devices']['kits'][kit]['fileNameInfo'] != '':
							src_path_info = join(raw_src_path, self.yaml['test']['devices']['kits'][kit]['fileNameInfo'])
							dict_info = dict()
							with open(src_path_info, 'rb') as infofile:
								for line in infofile:
									line = line.strip('\r\n')
									splitter = line.find(':')
									dict_info[line[:splitter]]= line[splitter+2:] # Accounting for the space
							   
							self.yaml['test']['devices']['kits'][kit]['info'] = dict_info
					
			
			# Create yaml with test description
			with open(join(_newpath, 'test_description.yaml'), 'w') as yaml_file:
				yaml.dump(self.yaml, yaml_file)
				
			print ('Test Creation Finished')