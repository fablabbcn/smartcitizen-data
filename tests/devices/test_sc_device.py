import pytest
import scdata as sc
from scdata._config import config

# Set basic configs
config._out_level = 'DEBUG'
config.data['strict_load'] = False

def test_device():
    id = 16549
    uuid = "d030cb8a-2c2a-429e-9f04-416888708193"
    min_date = '2023-07-29T09:00:06Z'
    blueprint = 'sc_21_station_module'

    # Make device
    device = sc.Device(descriptor = {'source': 'api',
                                     'id': id})

    load_status = device.load(options={'min_date': min_date})
    j = device.api_device.devicejson
    m = device.readings.index[0].tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    process_status = device.process()

    assert device.blueprint == blueprint, resp.text
    assert load_status == True, resp.text
    assert process_status == True, resp.text
    assert j['uuid'] == uuid, resp.text
    assert m == min_date
