import asyncio

from pandas import DataFrame
from smartcitizen_connector.tools import dict_fmerge, get_alphasense, get_pt_temp

from conftest import ADC_48_2, ADC_48_3, SCD30_CO2, SHT31_TEMP, load_json


def readings(index, **columns):
    return DataFrame({name: [value] * len(index) for name, value in columns.items()}, index=index)


def test_required_sensors_without_optional_sensor(make_device, blueprint):
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_2, ADC_48_3])

    # SCD30 is not in the device, ADC_48_2 is not used by the blueprint
    assert device.required_sensors == ['ADC_48_3', 'Sensirion SHT31 - Temperature']


def test_required_sensors_with_optional_sensor(make_device, blueprint):
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3, SCD30_CO2])

    assert device.required_sensors == ['ADC_48_3', 'Sensirion SCD30 - CO2', 'Sensirion SHT31 - Temperature']


def test_eager_channels_are_only_loaded_if_present(make_device, blueprint):
    device = make_device(blueprint, [ADC_48_3])

    assert device.required_sensors == ['ADC_48_3']


def test_required_sensors_ignores_values_that_are_not_sensors(make_device, blueprint):
    blueprint['channels'][2]['kwargs']['win_type'] = 'hann'
    device = make_device(blueprint, [SHT31_TEMP, SCD30_CO2])

    assert device.required_sensors == ['Sensirion SCD30 - CO2', 'Sensirion SHT31 - Temperature']


def test_load_requests_only_required_sensors(make_device, blueprint, index):
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_2, ADC_48_3],
                         readings(index, **{'Sensirion SHT31 - Temperature': 20.0, 'ADC_48_2': 0.1, 'ADC_48_3': 0.2}))
    device.options.channels = device.required_sensors

    assert asyncio.run(device.load()) is True
    assert device.handler.requested_channels == ['ADC_48_3', 'Sensirion SHT31 - Temperature']
    assert sorted(device.data.columns) == ['ADC_48_3', 'TEMP']


def test_process_skips_channels_without_sensors(make_device, blueprint, index):
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2)
    device.loaded = True

    assert device.process() is True
    assert 'EC_SENSOR_TEMP' in device.data
    assert 'NO2_WE' in device.data
    assert 'CO2_CLEAN' not in device.data
    assert 'CO2' not in device.data


def test_process_with_optional_sensor(make_device, blueprint, index):
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3, SCD30_CO2])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2, SCD30_CO2=500.0)
    device.loaded = True

    assert device.process() is True
    assert (device.data['CO2'] == 500.0).all()


def test_process_skips_sensor_without_readings(make_device, blueprint, index):
    # The device has the sensor, but it has no readings
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3, SCD30_CO2])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2)
    device.loaded = True

    assert device.process() is True
    assert 'CO2' not in device.data


def test_process_fails_when_sensor_request_failed(make_device, blueprint, index):
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3, SCD30_CO2], failed_sensors=['Sensirion SCD30 - CO2'])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2)
    device.loaded = True

    assert device.process() is False
    assert 'NO2_WE' in device.data
    assert 'CO2' not in device.data


def test_process_without_sensor_registry(make_device, blueprint, index):
    # As CSV files: no sensors, the data columns are all there is
    device = make_device(blueprint, [])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2)
    device.loaded = True

    assert device.process() is True
    assert 'NO2_WE' in device.data
    assert 'CO2' not in device.data


def test_required_sensors_with_sc_air_blueprint(make_device):
    ''' sc_air filled with hardware SCAS220097, as the connector does for a TwinAIR device '''
    blueprint = load_json('blueprints', 'sc_air.json')
    channels = {channel['name']: channel for channel in blueprint['channels']}
    filled = get_alphasense('AS_48_32', '212180226') + get_alphasense('AS_49_10', '214400458') + \
        get_pt_temp('PT_49_23', '10-002986')
    for item in filled:
        for name, value in item.items():
            channels[name]['kwargs'] = dict_fmerge(channels[name]['kwargs'], value['kwargs'])

    sensors = [{'id': 133 + i, 'name': f'ADC_48_{i}', 'description': 'ADC', 'unit': 'V'} for i in range(4)] + \
        [{'id': 138 + i, 'name': f'ADC_49_{i}', 'description': 'ADC', 'unit': 'V'} for i in range(4)] + \
        [SHT31_TEMP, {'id': 202, 'name': 'Sensirion SEN5X - TPS', 'description': 'TPS', 'unit': 'um'}]
    device = make_device(blueprint, sensors)

    assert device.required_sensors == ['ADC_48_2', 'ADC_48_3', 'ADC_49_0', 'ADC_49_1', 'ADC_49_2', 'ADC_49_3',
                                       'Sensirion SHT31 - Temperature']


def test_process_skips_sensor_with_empty_column(make_device, blueprint, index):
    # The column exists (e.g. data of another period) but has no readings here
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3, SCD30_CO2])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2, SCD30_CO2=float('nan'))
    device.loaded = True

    assert device.process() is True
    assert 'CO2' not in device.data


def test_process_survives_a_failing_channel(make_device, index):
    blueprint = {'channels': [
        {'name': 'BROKEN', 'function': 'poly_ts', 'kwargs': {'channels': ['TEMP'], 'coefficients': 'x'}},
        {'name': 'NO2_WE', 'function': 'channel_names', 'kwargs': {'channel': 'ADC_48_3'}},
    ]}
    device = make_device(blueprint, [SHT31_TEMP, ADC_48_3])
    device.data = readings(index, TEMP=20.0, ADC_48_3=0.2)
    device.loaded = True

    assert device.process() is False
    assert 'NO2_WE' in device.data
