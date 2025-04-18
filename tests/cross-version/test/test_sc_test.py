import pytest
import scdata as sc
from scdata.models import Metric
from scdata._config import config
import time
import asyncio
from scdata.tools.date import localise_date

# Set basic configs
config._out_level = 'DEBUG'

def test_sc_test():
    # Test couple of weirded format dates
    min_date = '2023-09-20 08:19:10-0700'
    now = time.localtime()
    devices_list=[16838, 16989]

    # Make test
    t = sc.Test(name=f'CHECK_{now.tm_year}-{now.tm_mon}-{now.tm_mday}',
        devices=[sc.Device(blueprint='sc_air',
        params=sc.APIParams(id=d),
        options=sc.DeviceOptions(min_date=min_date)) for d in devices_list],
            force_recreate=True)

    load_status = asyncio.run(t.load())
    metric = Metric(name='NOISE_A_SMOOTH',
                description='Basic smoothing calculation',
                function='rolling_avg',
                kwargs= {'name': ['NOISE_A'], 'window_size': 5}
               )
    t.get_device(16838).add_metric(metric)
    process_status = t.process()
    metric_in_df = 'NOISE_A_SMOOTH' in t.get_device(16838).data.columns

    assert process_status == True
    assert t.loaded == True
    assert metric_in_df == True
    for device in t.devices:
        assert (localise_date(min_date, 'UTC') < device.data.index[0])