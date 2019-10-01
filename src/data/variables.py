import os
import re

# Where are we?
rootDirectory = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

# Get URLSs from file (they are not necessarily secret) but we have them centralized in a txt
with open(os.path.join(rootDirectory, 'urls.txt')) as environment:
	for var in environment:
		if '#' not in var: 
			key = var.split('=')
			os.environ[key[0]] = re.sub('\n','',key[1])

SENSOR_NAMES_URL_21 = os.environ['SENSOR_NAMES_URL_21']
SENSOR_NAMES_URL_20 = os.environ['SENSOR_NAMES_URL_20']

API_BASE_URL = os.environ['API_BASE_URL']
API_KITS_URL = os.environ['API_KITS_URL']

## Other definitions
NAME_COMBINED_DATA = 'COMBINED_DEVICES'
