''' Processing per hardware version: each period with the channels (sensors) of its version '''
from datetime import datetime, timezone

import pandas as pd
import pytest

from fake_handler import FakeHandler

ADC = [{'id': 136, 'name': 'ADC_48_3', 'description': 'ADC', 'unit': 'V'},
       {'id': 139, 'name': 'ADC_49_1', 'description': 'ADC', 'unit': 'V'}]
BLUEPRINT = {'channels': [{'name': 'NO2_WE', 'function': 'channel_names', 'kwargs': {'channel': None}}]}


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def version(start, end, adc):
    return {'from_date': start, 'to_date': end,
            'channels': [{'name': 'NO2_WE', 'function': 'channel_names', 'kwargs': {'channel': adc}}]}


@pytest.fixture
def readings():
    index = pd.date_range('2024-11-30 23:00', '2025-01-01 02:00', freq='1h', tz='UTC')
    return pd.DataFrame({'ADC_48_3': 1.0, 'ADC_49_1': 2.0}, index=index)


def processed(device, readings, versions):
    device.versions = versions
    device.data = readings.copy()
    device.loaded = True
    return device.process()


def device_with_versions(make_device, versions):
    ''' Device channels are those of the latest version, as SCDevice gives them '''
    device = make_device({'channels': versions[-1]['channels']}, ADC)
    return device


def test_each_version_processes_its_period(make_device, readings):
    device = make_device(BLUEPRINT, ADC)

    versions = [version(utc(2024, 4, 1), utc(2025, 1, 1), 'ADC_48_3'), version(utc(2025, 1, 1), None, 'ADC_49_1')]
    device = device_with_versions(make_device, versions)

    assert processed(device, readings, versions)

    no2_we = device.data['NO2_WE']
    assert (no2_we[:'2024-12-31 23:00'] == 1.0).all()
    assert (no2_we['2025-01-01 00:00':] == 2.0).all()
    # Current channels stay those of the device
    assert device.channels[0].kwargs['channel'] == 'ADC_49_1'


def test_added_and_changed_channels_apply_to_all_periods(make_device, readings):
    versions = [version(utc(2024, 4, 1), utc(2025, 1, 1), 'ADC_48_3'), version(utc(2025, 1, 1), None, 'ADC_49_1')]
    device = device_with_versions(make_device, versions)
    device.add_channel({'name': 'ADC_SUM', 'function': 'poly_ts',
                        'kwargs': {'channels': ['ADC_48_3', 'ADC_49_1']}})

    assert processed(device, readings, versions)
    assert (device.data['ADC_SUM'] == 3.0).all()

    # Changed in code: same definition for all periods
    device.channels[0].kwargs = {'channel': 'ADC_48_3'}
    assert processed(device, readings, versions)
    assert (device.data['NO2_WE'] == 1.0).all()


def test_first_version_covers_earlier_data(make_device, readings):
    versions = [version(utc(2024, 12, 15), None, 'ADC_48_3')]
    device = device_with_versions(make_device, versions)

    assert processed(device, readings, versions)

    assert (device.data['NO2_WE'] == 1.0).all()


def test_gap_between_versions_is_not_processed(make_device, readings):
    versions = [version(utc(2024, 4, 1), utc(2024, 12, 1), 'ADC_48_3'), version(utc(2025, 1, 1), None, 'ADC_49_1')]
    device = device_with_versions(make_device, versions)

    assert processed(device, readings, versions)

    no2_we = device.data['NO2_WE']
    assert no2_we['2024-12-01':'2024-12-31 23:00'].isna().all()
    assert (no2_we[:'2024-11-30 23:00'] == 1.0).all()
    assert (no2_we['2025-01-01':] == 2.0).all()
    assert len(device.data) == len(readings)


def test_without_versions(make_device, readings):
    blueprint = {'channels': [{'name': 'NO2_WE', 'function': 'channel_names', 'kwargs': {'channel': 'ADC_49_1'}}]}
    device = make_device(blueprint, ADC)

    assert processed(device, readings, [])

    assert (device.data['NO2_WE'] == 2.0).all()


def test_required_sensors_of_all_versions(make_device):
    device = make_device(BLUEPRINT, ADC)
    device.versions = [version(utc(2024, 4, 1), utc(2025, 1, 1), 'ADC_48_3'), version(utc(2025, 1, 1), None, 'ADC_49_1')]

    assert device.required_sensors == ['ADC_48_3', 'ADC_49_1']


def test_versions_come_from_the_handler(make_device, monkeypatch):
    versions = [version(utc(2025, 1, 1), None, 'ADC_49_1')]
    monkeypatch.setattr(FakeHandler, 'blueprint_url', 'https://flows.smartcitizen.me/api/v1/blueprints/sc_air.json')
    monkeypatch.setattr(FakeHandler, 'properties', {'channels': versions[0]['channels']})
    monkeypatch.setattr(FakeHandler, 'channels_by_version', versions)

    device = make_device(BLUEPRINT, ADC)

    assert device.blueprint == 'sc_air'
    assert device.versions == versions


def test_sensor_added_in_a_later_version(make_device, readings):
    # The second sensor only has readings from 2025: its channel is skipped before
    readings.loc[:'2024-12-31 23:00', 'ADC_49_1'] = float('nan')
    versions = [version(utc(2024, 4, 1), utc(2025, 1, 1), None), version(utc(2025, 1, 1), None, 'ADC_49_1')]
    device = device_with_versions(make_device, versions)
    device.channels.append(device.channels[0].model_copy(update={
        'name': 'SUM', 'function': 'poly_ts', 'kwargs': {'channels': ['ADC_49_1']}}))

    assert processed(device, readings, versions)

    assert device.data['SUM'][:'2024-12-31 23:00'].isna().all()
    assert (device.data['SUM']['2025-01-01':] == 2.0).all()
