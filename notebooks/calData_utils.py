from IPython.display import display
import pandas as pd
from os.path import join, dirname
from os import getcwd

url_alpha = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-data/master/notebooks/caldata/alphasense.json'
url_mics = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-data/master/notebooks/caldata/mics.json'

def getCalData(_sensorType):
	if _sensorType == 'alphasense':
		url = url_alpha

	elif _sensorType == 'mics':
		url = url_mics

	try:
		calData = pd.read_json(url, orient='columns', lines = True)
	except:
		filePath = join(getcwd(), 'caldata/')
		print 'file://' + filePath + _sensorType + '.json'
		calData = pd.read_json('file://' + filePath + _sensorType + '.json', orient='columns', lines = True)
	
	calData.index = calData['Serial No']

	return calData