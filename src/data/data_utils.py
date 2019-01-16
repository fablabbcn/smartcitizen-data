import pandas as pd
from os.path import join, dirname, abspath
from os import getcwd, pardir

rootDirectory = abspath(join(abspath(join(getcwd(), pardir)), pardir))
filePath = join(rootDirectory, 'smartcitizen-iscape-models/CalibrationData/')

def getCalData(_sensorType, _calibrationDataPath):
	print (_calibrationDataPath)

	calData = pd.read_json('file://' + _calibrationDataPath + _sensorType + '.json', orient='columns', lines = True)
	calData.index = calData['Serial No']

	return calData