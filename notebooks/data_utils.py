from IPython.display import display
import pandas as pd
from os.path import join, dirname
from os import getcwd

def getCalData(_sensorType):

	url = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-models/master/CalibrationData/'

	try:
		calData = pd.read_json(url + _sensorType + '.json', orient='columns', lines = True)
	except:
		filePath = join(getcwd(), 'caldata/')
		# print 'file://' + filePath + _sensorType + '.json'
		calData = pd.read_json('file://' + filePath + _sensorType + '.json', orient='columns', lines = True)
	
	calData.index = calData['Serial No']

	return calData