#!/usr/bin/python

import scdata as sc
from scdata._config import config

config._out_level = 'DEBUG'

# Device id needs to be as str
device = sc.Device(descriptor = {'source': 'api', 'id': '4742'})

device.api_device.get_device_location()
device.api_device.get_device_sensors()
initial_kit_ID = device.api_device.get_kit_ID()

print ('Initial Kit ID')
print(device.api_device.kit_id)

# Check this script posts the data
device.api_device.kit_id = 29

device.api_device.post_kit_ID()
print (f'Kit ID after making it be {device.api_device.kit_id}')
print (device.api_device.get_kit_ID(update = True))

device.api_device.kit_id = initial_kit_ID
device.api_device.post_kit_ID()
check = device.api_device.get_kit_ID(update = True)
print ('Initial Kit ID (again, after reverting the last change)')
print(check)
