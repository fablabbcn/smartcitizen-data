from test_utils import *
from api_utils import *

class recordings:
	def __init__(self):
		self.readings = dict()

	def add_recording_CSV(self, reading_name, source_id, target_raster = '1Min', clean_na = True, clean_na_method = 'fill'):
		data = loadTest(source_id, target_raster, clean_na, clean_na_method)
		self.readings[reading_name] = dict()
		self.readings[reading_name] = data[reading_name]

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

	def del_recording(self, reading_name):
		if reading_name in self.readings.keys():
			self.readings.pop(reading_name)
		print ('Deleting', reading_name)

	def clear_recordings(self):
		self.readings.clear()
		print ('Clearing recordings')
