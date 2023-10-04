import pytest
import scdata as sc
from scdata._config import config
import time

# Set basic configs
config._out_level = 'DEBUG'
config.data['strict_load'] = False

def test_test():
    # Test couple of weirded format dates
    min_date = '2023-09-20 08:19:10-0700'
    max_date = '2023-09-30 08:19:12'

    now = time.localtime()
    # Make test
    t = sc.Test(f'CHECK_{now.tm_year}-{now.tm_mon}-{now.tm_mday}')
    t.add_devices_list(blueprint = 'sc_21_station_module',
        devices_list = [16609, "15618"])

    name = t.create()
    t.load(options={'min_date': min_date, 'max_date': max_date})

    load_status = t.loaded
    process_status = t.process()

    assert load_status == True, resp.text
    assert process_status == True, resp.text
    for device in t.devices:
        assert (sc.utils.localise_date(min_date, 'UTC') < t.devices[device].readings.index[0]), resp.text
        assert (sc.utils.localise_date(max_date, 'UTC') > t.devices[device].readings.index[0]), resp.text
