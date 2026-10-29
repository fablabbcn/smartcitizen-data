''' Blueprint of a device: the one of its hardware (by kind), the one given, or none '''
import asyncio

import pytest
from pandas import DataFrame, date_range

import scdata as sc
from scdata._config import config
from scdata.models import Blueprint

from fake_handler import FakeHandler
from conftest import SCD30_CO2, SHT31_TEMP

SOURCE = {'type': 'api', 'module': 'fake_handler', 'handler': 'FakeHandler'}


def test_the_kind_is_given_to_the_handler(monkeypatch, names):
    monkeypatch.setitem(config.names, 'FakeHandler', names)
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP])
    monkeypatch.setattr(FakeHandler, 'blueprint_name', 'sc_air_baseline')
    monkeypatch.setattr(FakeHandler, 'properties', {'meta': {'kind': 'long'}, 'channels': []})

    device = sc.Device(source=SOURCE, params=sc.APIParams(id=1), blueprint_kind='long')

    assert device.handler.blueprint_kind == 'long'
    assert (device.blueprint, device.blueprint_kind) == ('sc_air_baseline', 'long')


def test_the_kind_is_the_one_of_the_blueprint_used(make_device, monkeypatch):
    # Without a kind asked for, the process blueprint of the hardware
    monkeypatch.setattr(FakeHandler, 'blueprint_name', 'sc_air')
    monkeypatch.setattr(FakeHandler, 'properties', {'meta': {}, 'channels': []})
    device = make_device({'channels': []}, [SHT31_TEMP])

    assert (device.blueprint, device.blueprint_kind) == ('sc_air', 'process')

    device.use_blueprint('sc_air_baseline', {'meta': {'kind': 'long'}, 'channels': []})
    assert (device.blueprint, device.blueprint_kind) == ('sc_air_baseline', 'long')


def test_the_hardware_blueprint_is_used_before_the_one_given(make_device, monkeypatch):
    monkeypatch.setattr(FakeHandler, 'blueprint_name', 'sc_air')
    monkeypatch.setattr(FakeHandler, 'properties', {'channels': []})

    assert make_device({'channels': []}, [SHT31_TEMP]).blueprint == 'sc_air'


def test_without_a_blueprint_there_is_nothing_to_process(monkeypatch, names):
    # E.g. a hardware that is only backed up
    monkeypatch.setitem(config.names, 'FakeHandler', names)
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP])
    monkeypatch.setattr(FakeHandler, 'blueprint_name', None)

    device = sc.Device(source=SOURCE, params=sc.APIParams(id=1))

    assert device.blueprint is None and device.blueprint_kind is None and device.channels == []
    assert device.valid_for_processing is False


def test_without_a_blueprint_of_the_kind_asked_for(monkeypatch, names):
    monkeypatch.setitem(config.names, 'FakeHandler', names)
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP])
    monkeypatch.setattr(FakeHandler, 'blueprint_name', None)

    device = sc.Device(source=SOURCE, params=sc.APIParams(id=1), blueprint_kind='long')

    assert device.blueprint is None and device.blueprint_kind is None


def test_without_a_blueprint_the_sensors_are_loaded(monkeypatch, names):
    # E.g. a device without postprocessing
    monkeypatch.setitem(config.names, 'FakeHandler', names)
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP])
    monkeypatch.setattr(FakeHandler, 'blueprint_name', None)
    monkeypatch.setattr(FakeHandler, 'readings', DataFrame(
        {'Sensirion SHT31 - Temperature': [20.0, 21.0]}, index=date_range('2026-10-01', periods=2, freq='1min', tz='UTC')))
    device = sc.Device(source=SOURCE, params=sc.APIParams(id=1), options=sc.DeviceOptions(convert_units=False))

    assert asyncio.run(device.load()) is True
    # All the sensors, with their names
    assert device.handler.requested_channels == []
    assert list(device.data.columns) == ['TEMP']


def test_the_blueprint_given_overrides_the_hardware_one(make_device, monkeypatch):
    monkeypatch.setattr(FakeHandler, 'blueprint_name', 'sc_air')
    monkeypatch.setattr(FakeHandler, 'properties', {'channels': []})
    monkeypatch.setitem(config.blueprints, 'unit_test', {'channels': []})

    device = sc.Device(blueprint='unit_test', override_hardware_blueprint=True, source=SOURCE,
                       params=sc.APIParams(id=1))

    assert (device.blueprint, device.blueprint_kind) == ('unit_test', 'process')


# A device with or without a hardware (postprocessing), created with or without a blueprint

HARDWARE_BLUEPRINT = {'meta': {'kind': 'process'}, 'channels': [
    {'name': 'TEMP_F', 'function': 'poly_ts', 'kwargs': {'channels': ['TEMP'], 'coefficients': [1.8], 'extra_term': 32}}]}
GIVEN_BLUEPRINT = {'meta': {'kind': 'long'}, 'channels': [
    {'name': 'TEMP_K', 'function': 'poly_ts', 'kwargs': {'channels': ['TEMP'], 'coefficients': [1], 'extra_term': 273.15}}]}


@pytest.mark.parametrize('hardware, given, blueprint, kind, channel', [
    # The blueprint of the hardware, also when another one is given
    (True, False, 'sc_air', 'process', 'TEMP_F'),
    (True, True, 'sc_air', 'process', 'TEMP_F'),
    # Without a hardware: the one given, or none
    (False, True, 'unit_test', 'long', 'TEMP_K'),
    (False, False, None, None, None),
])
def test_with_or_without_hardware_and_blueprint(monkeypatch, names, hardware, given, blueprint, kind, channel):
    monkeypatch.setitem(config.names, 'FakeHandler', names)
    monkeypatch.setitem(config.blueprints, 'unit_test', Blueprint.model_validate(GIVEN_BLUEPRINT).model_dump())
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP, SCD30_CO2])
    monkeypatch.setattr(FakeHandler, 'readings', DataFrame(
        {'Sensirion SHT31 - Temperature': [20.0, 21.0], 'Sensirion SCD30 - CO2': [400.0, 410.0]},
        index=date_range('2026-10-01', periods=2, freq='1min', tz='UTC')))
    monkeypatch.setattr(FakeHandler, 'blueprint_name', 'sc_air' if hardware else None)
    monkeypatch.setattr(FakeHandler, 'properties', HARDWARE_BLUEPRINT if hardware else {})

    device = sc.Device(source=SOURCE, params=sc.APIParams(id=1), options=sc.DeviceOptions(convert_units=False),
                       **({'blueprint': 'unit_test'} if given else {}))

    assert (device.blueprint, device.blueprint_kind) == (blueprint, kind)
    assert [item.name for item in device.channels] == ([channel] if channel else [])
    # All the sensors are loaded, with their names
    assert asyncio.run(device.load()) is True
    assert sorted(device.data.columns) == ['SCD30_CO2', 'TEMP']
    assert device.process() is True
    assert sorted(device.data.columns) == sorted(['SCD30_CO2', 'TEMP'] + ([channel] if channel else []))


class ParamsOnlyHandler(FakeHandler):
    ''' An API handler that only takes params '''
    def __init__(self, params):
        super().__init__(params)


def test_handlers_that_only_take_params(monkeypatch, names):
    import fake_handler
    monkeypatch.setattr(fake_handler, 'ParamsOnlyHandler', ParamsOnlyHandler, raising=False)
    monkeypatch.setitem(config.names, 'ParamsOnlyHandler', names)
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP])
    monkeypatch.setattr(FakeHandler, 'blueprint_name', None)

    device = sc.Device(source={'type': 'api', 'module': 'fake_handler', 'handler': 'ParamsOnlyHandler'},
                       params=sc.APIParams(id=1))

    assert isinstance(device.handler, ParamsOnlyHandler)


@pytest.mark.parametrize('hardware, asked, override', [
    # A long blueprint asked for that the hardware does not have: the device is created again without one
    (None, 'long', False),
    ('sc_air_baseline', 'long', False),
    ('sc_air', None, True),
])
def test_the_test_keeps_how_the_device_was_created(monkeypatch, names, hardware, asked, override):
    from pydantic import TypeAdapter
    from typing import List

    monkeypatch.setitem(config.names, 'FakeHandler', names)
    monkeypatch.setitem(config.blueprints, 'unit_test', Blueprint.model_validate(GIVEN_BLUEPRINT).model_dump())
    monkeypatch.setattr(FakeHandler, 'sensors', [SHT31_TEMP])
    monkeypatch.setattr(FakeHandler, 'blueprint_name', hardware)
    monkeypatch.setattr(FakeHandler, 'properties', {'meta': {'kind': asked or 'process'}, 'channels': []})
    device = sc.Device(source=SOURCE, params=sc.APIParams(id=1), blueprint_kind=asked,
                       **({'blueprint': 'unit_test', 'override_hardware_blueprint': True} if override else {}))

    # What test.json stores of the device (without creating a test)
    from types import SimpleNamespace
    test = SimpleNamespace(name='TEST', path='', type='dev', options=sc.TestOptions(), devices=[device])
    stored = sc.Test.ser_model(test)['devices'][0]

    assert (stored['blueprint_kind'], stored['override_hardware_blueprint']) == (asked, override)
    again = TypeAdapter(List[sc.Device]).validate_python([stored])[0]
    assert (again.blueprint, again.blueprint_kind) == (device.blueprint, device.blueprint_kind)
