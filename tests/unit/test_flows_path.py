''' What flows does with a device: load, process, checks, post, backup to S3, and long processing
read from the backup. With the sc_air blueprints filled by the connector with the hardware of two kits '''
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter
from smartcitizen_connector import SCDevice
from smartcitizen_connector.models import HardwarePostprocessing

import scdata as sc
from scdata._config import config

from conftest import load_json
from fake_handler import FakeHandler

boto3 = pytest.importorskip('boto3')
moto = pytest.importorskip('moto')

SENSORS = ['ADC_48_0', 'ADC_48_1', 'ADC_48_2', 'ADC_48_3', 'ADC_49_0', 'ADC_49_1', 'ADC_49_2', 'ADC_49_3',
           'ADC_4A_0', 'ADC_4A_1', 'ADC_4A_2', 'ADC_4A_3', 'TEMP', 'SCD30_CO2']
# As the API names them: ec_sensor_temp looks for the SHT31 by its name
API_NAMES = {'TEMP': 'Sensirion SHT31 - Temperature'}
API_UNITS = {'TEMP': 'ºC'}
URL = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/master/blueprints/sc_air.json'

# Means of the channels with these readings: a pandas or numpy version that changes them fails
KITS = {
    # Two electrochemical sensors, and the PT1000 of the AFE for their temperature
    'SCAS210028': {
        'process': {'ASPT1000': -52.184088, 'EC_SENSOR_TEMP': -52.184088, 'NO2': -11.058876, 'O3': -356.238999,
                    'CO2': 485.209411},
        'long': {'CO2_BASELINE': 396.31466, 'CO2': 538.891928, 'NO2': 37.418095, 'O3': 0.001556},
    },
    # Four electrochemical sensors, temperature from the SHT31
    'SCAS210001': {
        'process': {'EC_SENSOR_TEMP': 19.983711, 'CO': 351.493898, 'NO2': 60.315641, 'SO2': 159.660376,
                    'O3': -122.478473, 'CO2': 485.209411},
        'long': {'CO2_BASELINE': 396.31466, 'CO2': 538.891928, 'CO': 20.236236, 'NO2': 36.762808, 'SO2': 32.48216,
                 'O3': 0.004287},
    },
}


def readings():
    ''' Three days every minute, with a gap of 30 minutes and an hour of outliers in the CO2 '''
    names = {name['name']: name for name in load_json('names', 'SCDevice.json')}
    rng = np.random.default_rng(0)
    index = pd.date_range('2026-01-01', periods=3 * 1440, freq='1min', tz='UTC')
    day = np.sin(np.arange(len(index)) * 2 * np.pi / 1440)
    columns = {}
    for name in SENSORS:
        if name.startswith('ADC'):
            columns[name] = 0.25 + 0.01 * day + rng.normal(0, 0.0005, len(index))
        elif name == 'TEMP':
            columns[name] = 20 + 5 * day + rng.normal(0, 0.1, len(index))
        else:
            columns[name] = 450 + 50 * day + rng.normal(0, 5, len(index))
    data = pd.DataFrame({API_NAMES.get(name, f'API {name}'): values for name, values in columns.items()}, index=index)
    data.iloc[100:130] = np.nan
    # find_outliers looks at the hourly mean: a single reading does not move it
    data.iloc[2000:2060, data.columns.get_loc('API SCD30_CO2')] = 3000.0
    sensors = [{'id': int(names[name]['id']), 'name': API_NAMES.get(name, f'API {name}'), 'description': name,
                'unit': API_UNITS.get(name, names[name]['unit'])} for name in SENSORS]
    return sensors, data


def connector_device(hardware):
    ''' SCDevice without network, to fill blueprints with the hardware as the connector does '''
    device = SCDevice.__new__(SCDevice)
    device.id = 1
    device.json = SimpleNamespace(last_reading_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for name in ('_channels', '_checks', '_exports', '_versions', '_filled_properties'):
        setattr(device, name, [])
    device._properties, device._blueprint = {}, None
    device._hardware_postprocessing = TypeAdapter(HardwarePostprocessing).validate_python(hardware)
    return device


@pytest.fixture(params=list(KITS))
def kit(request, monkeypatch):
    connector = connector_device(load_json('hardware', f'{request.param}.json'))
    connector.apply_blueprint(load_json('blueprints', 'sc_air.json'))
    sensors, data = readings()
    monkeypatch.setitem(config.names, 'FakeHandler', config.names['SCDevice'])
    for name, value in dict(sensors=sensors, readings=data, failed_sensors=[], blueprint_url=URL,
                            properties=connector.properties,
                            channels_by_version=connector.channels_by_version).items():
        monkeypatch.setattr(FakeHandler, name, value)
    return request.param, connector, data


def device():
    return sc.Device(blueprint='sc_air', source={'type': 'api', 'module': 'fake_handler', 'handler': 'FakeHandler'},
                     params=sc.APIParams(id=1))


def check(health, name):
    return next(item for item in health['checks'] if item['name'] == name)


def assert_means(data, means):
    for channel, mean in means.items():
        assert data[channel].notna().sum() > 0.9 * len(data), channel
        assert data[channel].mean() == pytest.approx(mean, rel=1e-6, abs=1e-6), channel


def test_process(kit):
    name = kit[0]
    d = device()
    d.options.channels = d.required_sensors

    assert asyncio.run(d.load())
    assert d.process()
    assert_means(d.data, KITS[name]['process'])

    assert d.health_checks()
    assert [item['name'] for item in d.health['checks']] == ['GAPS', 'IMPLAUSIBLE', 'FLAT', 'OUTLIERS']
    assert check(d.health, 'GAPS')['columns']['SCD30_CO2']['flagged'] == 30
    # Outliers from 09:20 to 10:19: the hourly mean is off from 09:37 to 11:00
    outlier = check(d.health, 'OUTLIERS')['columns']['SCD30_CO2']
    assert outlier['flagged'] == 84
    assert [[str(time) for time in interval] for interval in outlier['intervals']] == [
        ['2026-01-02T09:37:00+00:00', '2026-01-02T11:00:00+00:00']]

    assert asyncio.run(d.post(columns='channels', dry_run=True))


def test_backup_and_long_processing(kit, monkeypatch, tmp_path):
    name, connector, _ = kit
    d = device()
    d.options.channels = d.required_sensors
    assert asyncio.run(d.load())
    d.handler.json = SimpleNamespace(model_dump_json=lambda: '{}')
    for variable, value in dict(S3_DATA_BUCKET='bucket', AWS_DEFAULT_REGION='us-east-1', AWS_REGION='us-east-1',
                                AWS_ACCESS_KEY_ID='testing', AWS_SECRET_ACCESS_KEY='testing').items():
        monkeypatch.setenv(variable, value)
    backed_up = d.data.copy()

    with moto.mock_aws():
        s3 = boto3.client('s3')
        s3.create_bucket(Bucket='bucket')
        assert d.backup_to_storage(mode='append')
        # pyarrow reads S3 with its own client, which moto does not mock: read a local copy
        for item in s3.list_objects_v2(Bucket='bucket')['Contents']:
            target = tmp_path / item['Key']
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file('bucket', item['Key'], str(target))

    long = device()

    def apply_blueprint(blueprint):
        filled = connector.apply_blueprint(blueprint)
        long.handler.properties, long.handler.channels_by_version = connector.properties, connector.channels_by_version
        return filled

    long.handler.apply_blueprint = apply_blueprint
    assert long.use_blueprint('sc_air_baseline', load_json('blueprints', 'sc_air_baseline.json'))
    assert long.load_from_storage(channels=long.required_sensors, root=str(tmp_path))
    # The backup keeps the index as TIME, in nanoseconds (pandas 3 makes dates in microseconds)
    pd.testing.assert_frame_equal(long.data.set_axis(long.data.index.as_unit('ns')),
                                  backed_up[long.data.columns].set_axis(backed_up.index.as_unit('ns')),
                                  check_freq=False, check_names=False)

    assert long.process()
    assert_means(long.data, KITS[name]['long'])
    assert long.health_checks()
