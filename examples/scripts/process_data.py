#!/usr/bin/python

import scdata as sc

device = sc.Device(blueprint = 'sc_21_station_box', descriptor = {'source': 'api', 'id': '10751'})

device.api_device.post_info = dict()
device.api_device.post_info['hardware_id'] = "SCTEST"

device.__fill_metrics__()
device.load()
device.process()