''' Long processing: another blueprint for the same device, and data read from the backups '''
import pandas as pd
import pytest

from scdata.device.device import read_storage

ADC = [{'id': 136, 'name': 'ADC_48_3', 'description': 'ADC', 'unit': 'V'}]
SHORT = {'meta': {'kind': 'process'}, 'channels': [{'name': 'NO2_WE', 'function': 'channel_names',
                                                    'kwargs': {'channel': 'ADC_48_3'}}]}
LONG = {'meta': {'kind': 'long', 'window_days': 90}, 'checks': [{'name': 'GAPS', 'function': 'find_gaps'}],
        'channels': [{'name': 'DOUBLE', 'function': 'poly_ts',
                      'kwargs': {'channels': ['ADC_48_3'], 'coefficients': [2]}}]}


def test_use_blueprint(make_device):
    device = make_device(SHORT, ADC)
    device.data = pd.DataFrame({'ADC_48_3': [1.0, 2.0]}, index=pd.date_range('2026-01-01', periods=2, freq='1min', tz='UTC'))
    device.loaded = True

    assert device.use_blueprint('sc_air_baseline', LONG) is True

    assert device.blueprint == 'sc_air_baseline'
    assert device.meta == {'kind': 'long', 'window_days': 90}
    assert [channel.name for channel in device.channels] == ['DOUBLE']
    assert [check.name for check in device.checks] == ['GAPS']
    assert device.process()
    assert device.data['DOUBLE'].tolist() == [2.0, 4.0]


@pytest.fixture
def backup(tmp_path):
    ''' Two files appended by backup_to_storage, overlapping one reading '''
    folder = tmp_path / 'devices' / '1' / 'data'
    folder.mkdir(parents=True)
    for number, start in enumerate(['2026-01-01', '2026-01-10 23:59']):
        index = pd.date_range(start, periods=10 * 1440 if number == 0 else 1440, freq='1min', tz='UTC')
        frame = pd.DataFrame({'TEMP': float(number), 'HUM': 50.0, 'TIME': index})
        frame.to_parquet(folder / f'part-{number}.parquet', index=False)
    return tmp_path


def test_read_storage_with_period_and_columns(backup):
    data = read_storage(str(backup / 'devices' / '1' / 'data'), min_date='2026-01-10', max_date='2026-01-11 00:00',
                        channels=['TEMP', 'MISSING'])

    assert list(data.columns) == ['TEMP']
    assert data.index.min() == pd.Timestamp('2026-01-10', tz='UTC')
    assert data.index.max() == pd.Timestamp('2026-01-11', tz='UTC')
    assert data.index.is_unique and len(data) == 1441
    # The overlapping reading comes from the first file
    assert data.loc['2026-01-10 23:59', 'TEMP'] == 0.0


@pytest.mark.parametrize('units', [('ns', 'us'), ('us', 'ns')])
def test_read_storage_with_times_in_ns_and_us(tmp_path, units):
    # Backups written with pandas 2 (ns) and pandas 3 (us) in the same folder
    for number, unit in enumerate(units):
        index = pd.date_range(f'2026-01-0{number + 1}', periods=3, freq='1min', tz='UTC').as_unit(unit)
        pd.DataFrame({'TEMP': float(number), 'TIME': index}).to_parquet(tmp_path / f'part-{number}.parquet', index=False)

    data = read_storage(str(tmp_path), min_date='2026-01-01 00:01')

    assert len(data) == 5 and data.index.is_monotonic_increasing
    assert data.index.min() == pd.Timestamp('2026-01-01 00:01', tz='UTC')
    assert data.index.max() == pd.Timestamp('2026-01-02 00:02', tz='UTC')


def test_load_from_storage(make_device, backup):
    device = make_device(SHORT, ADC)

    assert device.load_from_storage(root=str(backup), min_date='2026-01-11 12:00')

    # The second file ends at 2026-01-11 23:58
    assert device.loaded and len(device.data) == 719 and set(device.data.columns) == {'TEMP', 'HUM'}
    assert not device.load_from_storage(root=str(backup / 'nowhere'))


def test_load_from_storage_with_the_sensors_of_the_api(make_device, backup):
    # required_sensors gives API names: the backups keep the converted ones (TEMP)
    from conftest import SHT31_TEMP
    device = make_device(SHORT, ADC + [SHT31_TEMP])

    assert device.load_from_storage(root=str(backup), min_date='2026-01-11 12:00',
                                    channels=['Sensirion SHT31 - Temperature', 'ADC_48_3'])

    assert list(device.data.columns) == ['TEMP']


def test_load_from_storage_without_pyarrow(make_device, backup, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name.startswith('pyarrow'):
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_pyarrow)
    device = make_device(SHORT, ADC)

    assert device.load_from_storage(root=str(backup)) is False


def test_qc_data_is_only_read_from_s3(make_device, backup):
    device = make_device(SHORT, ADC)

    # The data loads from the local root; qc data is not looked for elsewhere
    assert device.load_from_storage(root=str(backup), load_qc_data=True) is True
    assert device.qc_data.empty


def test_s3_uses_the_region_of_the_bucket(monkeypatch):
    # A region in the environment that is not the one of the bucket made reads fail
    from pyarrow import fs
    import pyarrow.dataset as ds
    import scdata.device.device as device

    monkeypatch.setenv('AWS_REGION', 'eu-south-1')
    monkeypatch.setattr(device, '_bucket_regions', {})
    asked, opened = [], []
    monkeypatch.setattr(fs, 'resolve_s3_region', lambda bucket: asked.append(bucket) or 'eu-west-1')
    monkeypatch.setattr(ds, 'dataset', lambda location, filesystem, format: opened.append((location, filesystem.region))
                        or (_ for _ in ()).throw(FileNotFoundError(location)))

    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            read_storage('s3://bucket/devices/1/data')

    assert opened == [('bucket/devices/1/data', 'eu-west-1')] * 2
    assert asked == ['bucket']


def test_bucket_region_from_the_environment_if_s3_cannot_be_asked(monkeypatch):
    from pyarrow import fs
    import scdata.device.device as device

    monkeypatch.setenv('AWS_REGION', 'eu-west-1')
    monkeypatch.setattr(device, '_bucket_regions', {})

    def unreachable(bucket):
        raise OSError('unreachable')

    monkeypatch.setattr(fs, 'resolve_s3_region', unreachable)
    assert device.bucket_region('bucket') == 'eu-west-1'
    # Asked again next time
    monkeypatch.setattr(fs, 'resolve_s3_region', lambda bucket: 'eu-central-1')
    assert device.bucket_region('bucket') == 'eu-central-1'
