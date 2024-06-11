import pytest
import scdata as sc
from scdata._config import config
import asyncio
import os

# Set basic configs
config._log_level = 'DEBUG'

def test_csv_device_all():
    id = 16838
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..', 'scdata/tools/interim/example.csv')
    frequency = '5Min'

    min_date = '2020-09-02T15:35:19Z'
    blueprint = 'sc_air'

    d = sc.Device(
        blueprint=blueprint,
        source={'type':'csv',
                'handler': 'CSVHandler',
                'module': 'scdata.io.device_file'},
        params=sc.CSVParams(id=id,
            path=path,
            timezone='Europe/Madrid')
        )

    # Make device
    d.options.frequency = frequency
    d.options.min_date = min_date
    load_status = asyncio.run(d.load())

    m = d.data.index[0].tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    process_status = d.process()

    assert d.blueprint == blueprint, resp.text
    assert load_status == True, resp.text
    assert process_status == True, resp.text
    assert m == min_date
