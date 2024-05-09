#!/usr/bin/python

from smartcitizen_connector import SCDevice
from scdata._config import config

# Set verbose level
config.log_level = 'DEBUG'

# Device id needs to be as str
device = SCDevice(10972)
device.options.min_date = None #Don't trim min_date
device.options.max_date = None #Don't trim max_date

# Load
await data = device.g(min_date = None, max_date = None, frequency = '1Min', clean_na = None);

print (data)
