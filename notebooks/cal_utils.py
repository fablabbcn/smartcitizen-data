import pandas as pd

url = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-iscape-data/internal_dev/calData/AlphaSense.json'

def alphasense():

	alpha_calData = pd.read_json(url, orient='columns', lines = True)
	alpha_calData.index = alpha_calData['Serial No']
	return alpha_calData 