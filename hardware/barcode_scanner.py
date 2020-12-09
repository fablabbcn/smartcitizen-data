import re
import sys
import argparse
import json
import usb.core
import usb.util
from os.path import dirname, join, realpath

attempts = 10
data = None

def init_device():
    # find the USB device
    device = usb.core.find(idVendor = 59473)

    # use the first/default configuration
    device.set_configuration()
    # first endpoint
    endpoint = device[0][(0,0)][0]

    return device, endpoint

def read(_device, _endpoint):
    while data is None and attempts > 0:
      try:
          data = _device.read(_endpoint.bEndpointAddress, _endpoint.wMaxPacketSize)
      except usb.core.USBError as e:
          data = None
          if e.args == ('Operation timed out',):
              attempts -= 1
              continue

def input_data(json_file, hardware_id, number_sensors = 4):

    print ("Initialising device")
    device, endpoint = init_device()

    devices = list()
    while True:
        data = device.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize)
        if data is not None: 
            datac = data.strip('\n')[1:]
            devices.append(datac)

    print (devices)

    # TODO save json

if __name__ == "__main__":
    ldir = dirname(realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default = "hardware.json", help="I/O hardware filename, including extension")
    parser.add_argument("--hardware_id", "-hid", help="Final name of time index")
    parser.add_argument("--numsensors", "-n", default = 4, help="Number of sensors")
    
    args = parser.parse_args()
    input_data(join(ldir, args.file), args.hardware_id, args.numsensors)
