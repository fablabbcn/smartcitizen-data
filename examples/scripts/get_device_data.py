#!/usr/bin/python

from scdata.io.device_api import ScApiDevice
from scdata._config import config

# Set verbose level
config._out_level = 'DEBUG'

# Device id needs to be as str
device = ScApiDevice('10972')
device.get_device_lat_long()
device.get_device_sensors()

# Load
data = device.get_device_data(min_date = None, max_date = None, frequency = '1Min', clean_na = None);

print (data)
