import asyncio
from os import listdir
from os.path import getmtime, join

from pandas import DataFrame

import scdata as sc
from scdata._config import config

from conftest import SHT31_TEMP
from fake_handler import FakeHandler


def test_the_cache_is_only_written_with_new_data_and_in_the_format_of_the_test(make_device, index, monkeypatch, tmp_path):
    monkeypatch.setitem(config.paths, 'processed', str(tmp_path))
    readings = DataFrame({SHT31_TEMP['name']: range(len(index))}, index=index.rename('TIME'), dtype=float)
    device = make_device({'channels': []}, [SHT31_TEMP], readings)
    test = sc.Test(name='cache', devices=[device])
    cached = join(test.path, 'cached')

    assert asyncio.run(test.load())
    assert listdir(cached) == ['1.csv']
    written = getmtime(join(cached, '1.csv'))

    # Nothing new from the source: the data comes from the cache, which is kept
    monkeypatch.setattr(FakeHandler, 'readings', DataFrame())
    assert asyncio.run(test.load())
    assert getmtime(join(cached, '1.csv')) == written
    assert len(device.data) == len(index)

    # Moved to parquet, from the cache in csv
    test.options.cache_format = 'parquet'
    assert asyncio.run(test.load())
    assert listdir(cached) == ['1.parquet']

    assert asyncio.run(test.load())
    assert device.data.iloc[:, 0].tolist() == list(range(len(index)))
    assert str(device.data.index.tz) == 'UTC'
