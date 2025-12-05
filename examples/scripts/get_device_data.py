#!/usr/bin/python

import scdata as sc
from scdata._config import config
import asyncio

# Set verbose level
config.log_level = 'DEBUG'

device = sc.Device(blueprint='sc_air',
                               params=sc.APIParams(id=16784))
device.options.min_date = None #Don't trim min_date
device.options.max_date = None #Don't trim max_date
device.options.frequency = '1Min' # Use this to change the sample frequency for the request
print (device.json)

# Load
asyncio.run(device.load())

print (device.json)
print (device.data)
