import pytest
import scdata as sc
from scdata._config import config
import asyncio

# Set basic configs
config._log_level = 'DEBUG'

def test_sc_device_all():
    id = 16838
    frequency = '1Min'
    uuid = "80e684e5-359f-4755-aec9-30fc0c84415f"
    min_date = '2022-09-10T00:00:23Z'
    blueprint = 'sc_air'
    limit = 500
    channels = ['Sensirion SHT31 - Temperature', 'Sensirion SHT31 - Humidity', 'ICS43432 - Noise']

    d = sc.Device(blueprint=blueprint,
        params=sc.APIParams(id=id),
        options=sc.DeviceOptions(
            min_date=min_date,
            frequency=frequency,
            channels=channels,
            limit=limit)
        )

    load_status = asyncio.run(d.load())

    j = d.handler.json
    m = d.data.index[0].tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    s = d.data.shape
    process_status = d.process()

    assert d.blueprint == blueprint
    # assert s == (500, 3) #TODO - add when reading on staging
    assert load_status == True
    assert process_status == False #Force this to False
    assert j.uuid == uuid
    assert m == min_date
