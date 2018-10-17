from IPython.display import display
import pandas as pd
from os.path import join, dirname, abspath
from os import getcwd, pardir

url = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-models/master/CalibrationData/'

rootDirectory = abspath(join(abspath(join(getcwd(), pardir)), pardir))
filePath = join(rootDirectory, 'smartcitizen-iscape-models/CalibrationData/')

def getCalData(_sensorType):

	try:
		calData = pd.read_json(url + _sensorType + '.json', orient='columns', lines = True)
	except:
		# print rootDirectory
		# print filePath
		# print 'file://' + filePath + _sensorType + '.json'
		calData = pd.read_json('file://' + filePath + _sensorType + '.json', orient='columns', lines = True)
	
	calData.index = calData['Serial No']

	return calData