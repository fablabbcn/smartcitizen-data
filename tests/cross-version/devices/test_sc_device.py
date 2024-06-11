import pytest
import scdata as sc
from scdata._config import config
import asyncio

# Set basic configs
config._log_level = 'DEBUG'

def test_sc_device():
    id = 16838
    frequency = '1Min'
    uuid = "80e684e5-359f-4755-aec9-30fc0c84415f"
    min_date = '2022-09-10T00:00:00Z'
    blueprint = 'sc_air'

    d = sc.Device(blueprint=blueprint,
        params=sc.APIParams(id=id),
        options=sc.DeviceOptions(
            min_date=min_date,
            frequency=frequency)
        )

    load_status = asyncio.run(d.load())

    j = d.handler.json
    m = d.data.index[0].tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    process_status = d.process()

    assert d.blueprint == blueprint, resp.text
    assert load_status == True, resp.text
    assert process_status == True, resp.text
    assert j.uuid == uuid, resp.text
    assert m == min_date
