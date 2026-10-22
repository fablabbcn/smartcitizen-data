''' Creating, saving and opening tests: in the scdata cache by default, or in a custom folder '''
import asyncio
import builtins
import json
from datetime import datetime
from os.path import exists, join

import pytest
from pandas import DataFrame, Timedelta, Timestamp

import scdata as sc
from scdata._config import config

from conftest import SCD30_CO2, SHT31_TEMP


@pytest.fixture(autouse=True)
def paths(monkeypatch, tmp_path):
    ''' Keeps the scdata data folder out of the user cache '''
    monkeypatch.setitem(config.paths, 'data', str(tmp_path / 'cache'))
    monkeypatch.setitem(config.paths, 'processed', str(tmp_path / 'cache' / 'processed'))
    (tmp_path / 'cache' / 'processed').mkdir(parents=True)
    return tmp_path


@pytest.fixture
def answers(monkeypatch):
    ''' Answers the input prompts in order '''
    def set_answers(*values):
        values = iter(values)
        monkeypatch.setattr(builtins, 'input', lambda prompt='': next(values))
    return set_answers


@pytest.fixture
def device(make_device):
    return make_device({'channels': []}, [SHT31_TEMP])


def full_name(name):
    now = datetime.now()
    return f'{now.year}_{str(now.month).zfill(2)}_{name}'


def cache_path(name):
    now = datetime.now()
    return join(config.paths['processed'], str(now.year), str(now.month).zfill(2), full_name(name))


def stored_meta(test):
    with open(join(test.path, 'test.json')) as file:
        return json.load(file)['meta']


def test_create_in_memory():
    test = sc.Test(name='MEMORY')

    assert test.name == full_name('MEMORY')
    assert test.path == ''
    assert sc.tests().empty


def test_name_with_date_prefix():
    assert sc.Test(name='2025_03_CLIENT').name == '2025_03_CLIENT'


def test_save_in_cache():
    test = sc.Test(name='DEFAULT').save()

    assert test.path == cache_path('DEFAULT')
    assert stored_meta(test)['path'] == test.path
    # Tests in the cache are not registered
    assert not exists(join(config.paths['data'], 'tests.json'))


def test_save_in_prefix_month():
    test = sc.Test(name='2025_03_OLD').save()

    assert test.path == join(config.paths['processed'], '2025', '03', '2025_03_OLD')


def test_save_in_root(paths, monkeypatch):
    monkeypatch.chdir(paths)
    test = sc.Test(name='CUSTOM').save(root='tests')

    assert test.path == str(paths / 'tests' / full_name('CUSTOM'))
    assert stored_meta(test)['path'] == test.path


def test_save_existing():
    sc.Test(name='TWICE').save()

    with pytest.raises(FileExistsError):
        sc.Test(name='TWICE').save()
    assert sc.Test(name='TWICE', type='other').save(overwrite=True).path == cache_path('TWICE')
    assert sc.Test.open('TWICE').type == 'other'


def test_save_existing_elsewhere(paths):
    sc.Test(name='ELSEWHERE').save(root=str(paths / 'tests'))

    # The same name in the cache would make it ambiguous
    with pytest.raises(FileExistsError):
        sc.Test(name='ELSEWHERE').save()


def test_save_again_updates(device):
    test = sc.Test(name='UPDATE').save()
    test.devices = [device]
    test.save()

    assert [d.id for d in sc.Test.open('UPDATE').devices] == [1]


def test_open(device):
    sc.Test(name='OPEN', devices=[device]).save()

    for name in ['OPEN', full_name('OPEN'), cache_path('OPEN'), join(cache_path('OPEN'), 'test.json')]:
        test = sc.Test.open(name)
        assert test.name == full_name('OPEN')
        assert test.path == cache_path('OPEN')
        assert [d.id for d in test.devices] == [1]


def test_device_options_stored(device):
    device.options.min_date = '2026-01-01'
    device.options.frequency = '5Min'
    device.options.channels = ['TEMP']
    sc.Test(name='OPTIONS', devices=[device]).save()

    options = sc.Test.open('OPTIONS').devices[0].options
    assert options.min_date == '2026-01-01'
    assert options.frequency == '5Min'
    assert options.channels == ['TEMP']
    assert options.convert_units is False


def test_device_options_missing(device):
    # Tests saved before the device options were stored
    test = sc.Test(name='NO_OPTIONS', devices=[device]).save()
    with open(join(test.path, 'test.json')) as file:
        stored = json.load(file)
    del stored['devices'][0]['options']
    with open(join(test.path, 'test.json'), 'w') as file:
        json.dump(stored, file)

    assert sc.Test.open('NO_OPTIONS').devices[0].options == sc.DeviceOptions()


def test_override_url_blueprint_stored(monkeypatch, make_device):
    from fake_handler import FakeHandler
    import scdata.device.device as device_module

    device = make_device({'channels': []}, [SHT31_TEMP])
    device.override_url_blueprint = True
    sc.Test(name='OVERRIDE', devices=[device]).save()

    # The blueprint url is not checked when it is overridden
    monkeypatch.setattr(FakeHandler, 'blueprint_url', 'https://example.com/blueprints/other.json')
    def no_url_check(url):
        raise AssertionError('The blueprint url should not be checked')
    monkeypatch.setattr(device_module, 'url_checker', no_url_check)

    opened = sc.Test.open('OVERRIDE').devices[0]
    assert opened.override_url_blueprint is True
    assert opened.blueprint == 'unit_test'


def test_open_exact_name():
    sc.Test(name='CLIENT_A_2').save()

    with pytest.raises(FileNotFoundError):
        sc.Test.open('CLIENT_A')
    with pytest.raises(FileNotFoundError):
        sc.Test.open('CLIENT_A_')


def test_open_ambiguous():
    sc.Test(name='2025_03_SAME').save()
    sc.Test(name='2025_04_SAME').save()

    with pytest.raises(ValueError, match='Several tests'):
        sc.Test.open('SAME')
    assert sc.Test.open('2025_04_SAME').name == '2025_04_SAME'


def test_open_registered(paths):
    root = paths / 'tests'
    sc.Test(name='REGISTERED').save(root=str(root))

    # Found without the root
    assert sc.Test.open('REGISTERED').path == str(root / full_name('REGISTERED'))


def test_open_in_root(paths):
    # Copied from somewhere else: not registered
    test = sc.Test(name='COPIED').save(root=str(paths / 'tests'))
    (paths / 'cache' / 'tests.json').unlink()

    with pytest.raises(FileNotFoundError):
        sc.Test.open('COPIED')
    assert sc.Test.open('COPIED', root=str(paths / 'tests')).path == test.path


def test_open_moved(paths):
    root = paths / 'tests'
    sc.Test(name='MOVED').save(root=str(root))
    moved = paths / 'other'
    root.rename(moved)

    test = sc.Test.open(str(moved / full_name('MOVED')))
    assert test.path == str(moved / full_name('MOVED'))
    assert stored_meta(test)['path'] == test.path
    # Registered in its new folder
    assert sc.Test.open('MOVED').path == test.path


def test_open_or_create(paths, device, make_device, caplog):
    root = str(paths / 'tests')
    created = sc.Test.open_or_create('BOTH', devices=[device], root=root)
    assert created.path == str(paths / 'tests' / full_name('BOTH'))

    opened = sc.Test.open_or_create('BOTH', devices=[device], root=root)
    assert opened.path == created.path
    assert 'other devices' not in caplog.text

    other = make_device({'channels': []}, [SCD30_CO2])
    other.params.id = 2
    other.params_parsed.id = 2
    opened = sc.Test.open_or_create('BOTH', devices=[other])
    assert [d.id for d in opened.devices] == [1]
    assert 'other devices' in caplog.text


def test_tests(paths, device):
    sc.Test(name='CLIENT_A', devices=[device]).save()
    sc.Test(name='CLIENT_B').save(root=str(paths / 'tests'))
    sc.Test(name='OTHER').save()

    listed = sc.tests()
    assert sorted(listed['name']) == sorted(full_name(n) for n in ['CLIENT_A', 'CLIENT_B', 'OTHER'])
    assert sorted(sc.tests('client')['name']) == [full_name('CLIENT_A'), full_name('CLIENT_B')]
    assert listed.set_index('name').loc[full_name('CLIENT_A'), 'devices'] == 1


def test_save_ask_cache(answers):
    answers('')
    assert sc.Test(name='ASK_CACHE').save(root='ask').path == cache_path('ASK_CACHE')


def test_save_ask_current_folder(paths, monkeypatch, answers):
    monkeypatch.chdir(paths)
    answers('2')
    assert sc.Test(name='ASK_CWD').save(root='ask').path == str(paths / full_name('ASK_CWD'))


def test_save_ask_other_folder(paths, answers):
    answers('4', '3', str(paths / 'other'))
    test = sc.Test(name='ASK_OTHER').save(root='ask')

    assert test.path == str(paths / 'other' / full_name('ASK_OTHER'))


def test_save_ask_without_input(monkeypatch):
    def no_input(prompt=''):
        raise EOFError
    monkeypatch.setattr(builtins, 'input', no_input)

    assert sc.Test(name='ASK_NONE').save(root='ask').path == cache_path('ASK_NONE')


def test_open_or_create_ask_only_when_new(paths, answers):
    answers('3', str(paths / 'tests'))
    created = sc.Test.open_or_create('ASK_ONCE', root='ask')
    answers()

    # No input is needed: the test exists
    assert sc.Test.open_or_create('ASK_ONCE', root='ask').path == created.path


def test_not_saved():
    test = sc.Test(name='NOT_SAVED')

    assert test.purge() is False
    assert test.compress() is False
    assert test.to_csv() is False


def test_compress(paths):
    test = sc.Test(name='ZIP').save(root=str(paths / 'tests'))

    assert test.compress() == str(paths / 'tests' / f"{full_name('ZIP')}_full.zip")


# A reopened test reads the cached data first, and requests only what the cache doesn't have

def loaded_test(make_device, index, min_date=None, max_date=None):
    # Named as the API data
    readings = DataFrame({SHT31_TEMP['name']: range(len(index)), SCD30_CO2['name']: 400},
                         index=index.rename('TIME'), dtype=float)
    device = make_device({'channels': []}, [SHT31_TEMP, SCD30_CO2], readings=readings)
    device.options.channels = [SHT31_TEMP['name']]
    device.options.min_date = min_date
    device.options.max_date = max_date
    test = sc.Test(name='CACHED', devices=[device]).save()
    assert asyncio.run(test.load())
    assert exists(join(test.path, 'cached', '1.csv'))
    return test


def reload(min_date='stored', max_date='stored', channels='stored'):
    test = sc.Test.open('CACHED')
    device = test.devices[0]
    if min_date != 'stored': device.options.min_date = min_date
    if max_date != 'stored': device.options.max_date = max_date
    if channels != 'stored': device.options.channels = channels
    assert asyncio.run(test.load())
    return device


def dates(requests):
    return [tuple(None if date is None else Timestamp(date, tz='UTC') if Timestamp(date).tzinfo is None
                  else Timestamp(date).tz_convert('UTC') for date in request) for request in requests]


def test_reopened_cache_covers_dates(make_device, index):
    loaded_test(make_device, index, '2026-01-01 00:00', '2026-01-01 00:09')
    device = reload()

    # No request: the cache reaches max_date
    assert device.handler.requests == []
    assert len(device.data) == 10


def test_reopened_requests_new_data(make_device, index):
    loaded_test(make_device, index)
    device = reload()

    assert dates(device.handler.requests) == [(index[-1] + Timedelta('1min'), None)]
    assert len(device.data) == 10


def test_reopened_with_earlier_min_date(make_device, index):
    loaded_test(make_device, index, '2026-01-01 00:05', '2026-01-01 00:09')
    device = reload(min_date='2026-01-01 00:00')

    # Only the period before the cache
    assert dates(device.handler.requests) == [(index[0], index[4])]
    assert len(device.data) == 10
    assert device.data.index.is_monotonic_increasing


def test_reopened_with_narrower_dates(make_device, index):
    loaded_test(make_device, index, '2026-01-01 00:00', '2026-01-01 00:09')
    device = reload(min_date='2026-01-01 00:03', max_date='2026-01-01 00:06')

    # Only the cached data between the dates
    assert device.handler.requests == []
    assert list(device.data.index) == list(index[3:7])


def test_unreadable_cache(make_device, index, caplog):
    test = loaded_test(make_device, index)
    with open(join(test.path, 'cached', '1.csv'), 'w') as file:
        file.write(',TEMP\n2026-01-01 00:00:00+00:00,0.0\n')
    device = reload()

    # All the data is requested again
    assert dates(device.handler.requests) == [(None, None)]
    assert 'could not be read' in caplog.text
    assert len(device.data) == 10


def test_reopened_with_a_new_channel(make_device, index):
    loaded_test(make_device, index, '2026-01-01 00:00', '2026-01-01 00:09')
    device = reload(channels=[SHT31_TEMP['name'], SCD30_CO2['name']])

    # Only the new channel, over the cached dates
    assert dates([request[:2] for request in device.handler.requests]) == [(index[0], index[-1])]
    assert device.handler.requested_channels == [SCD30_CO2['name']]
    assert len(device.data) == 10
    assert len(device.data.columns) == 2
    assert not device.data.isna().any().any()
