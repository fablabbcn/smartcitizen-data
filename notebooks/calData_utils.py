from IPython.display import display
import pandas as pd

url_alpha = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-data/master/calData/AlphaSense.json'
url_mics = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-data/master/calData/mics.json'

def getCalData(_sensorType):
	if _sensorType == 'alphasense':
		url = url_alpha
	elif _sensorType == 'mics':
		url = url_mics

	calData = pd.read_json(url, orient='columns', lines = True)
	calData.index = calData['Serial No']
	return calData 