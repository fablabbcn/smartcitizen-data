import json
from os.path import dirname, join, realpath
from typing import List

import pytest
from pandas import DataFrame, date_range
from pydantic import TypeAdapter

import scdata as sc
from scdata._config import config
from scdata.models import Blueprint, Name

from fake_handler import FakeHandler

ROOT = join(dirname(realpath(__file__)), '..', '..')

# API sensors (name, id) as returned by the platform
SHT31_TEMP = {'id': 55, 'name': 'Sensirion SHT31 - Temperature', 'description': 'Temperature', 'unit': 'degC'}
SCD30_CO2 = {'id': 158, 'name': 'Sensirion SCD30 - CO2', 'description': 'CO2', 'unit': 'ppm'}
ADC_48_2 = {'id': 135, 'name': 'ADC_48_2', 'description': 'ADC', 'unit': 'V'}
ADC_48_3 = {'id': 136, 'name': 'ADC_48_3', 'description': 'ADC', 'unit': 'V'}


def load_json(*path):
    with open(join(ROOT, *path)) as file:
        return json.load(file)


@pytest.fixture
def names():
    return TypeAdapter(List[Name]).validate_python(load_json('names', 'SCDevice.json'))


@pytest.fixture
def make_device(monkeypatch, names):
    ''' Creates an offline Device with the given blueprint, sensors and readings '''
    monkeypatch.setitem(config.names, 'FakeHandler', names)

    def make(blueprint, sensors, readings=None, failed_sensors=None):
        monkeypatch.setitem(config.blueprints, 'unit_test', Blueprint.model_validate(blueprint).model_dump())
        monkeypatch.setattr(FakeHandler, 'sensors', sensors)
        monkeypatch.setattr(FakeHandler, 'readings', readings if readings is not None else DataFrame())
        monkeypatch.setattr(FakeHandler, 'failed_sensors', failed_sensors or [])
        return sc.Device(blueprint='unit_test',
                         source={'type': 'api', 'module': 'fake_handler', 'handler': 'FakeHandler'},
                         params=sc.APIParams(id=1),
                         options=sc.DeviceOptions(convert_units=False))

    return make


@pytest.fixture
def index():
    return date_range('2026-01-01', periods=10, freq='1min', tz='UTC')


@pytest.fixture
def blueprint():
    ''' Blueprint with an optional sensor chain (CO2) and an always available channel '''
    return {
        'channels': [
            {'name': 'EC_SENSOR_TEMP', 'function': 'ec_sensor_temp',
             'kwargs': {'priority': 'ASPT1000',
                        'eager_channels': ['Sensirion SHT31 - Temperature', 'Sensirion SHT35 - Temperature']}},
            {'name': 'NO2_WE', 'function': 'channel_names', 'kwargs': {'channel': 'ADC_48_3'}},
            {'name': 'CO2_CLEAN', 'function': 'clean_ts',
             'kwargs': {'name': 'SCD30_CO2', 'limits': [300, 9500], 'window': 1}},
            {'name': 'CO2', 'function': 'poly_ts', 'depends_on': ['CO2_CLEAN'],
             'kwargs': {'channels': ['SCD30_CO2', 'CO2_CLEAN'], 'coefficients': [1, 0]}},
        ]
    }
