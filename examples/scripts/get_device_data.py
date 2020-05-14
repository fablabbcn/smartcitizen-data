from scdata.io.read_api import ScApiDevice
from os.path import dirname, join

# Device id needs to be as str
device = ScApiDevice('10972')
device.get_device_location()
device.get_device_sensors()

# Load
data = device.get_device_data(start_date = None, end_date = None, frequency = '1Min', clean_na = None);

print (data)