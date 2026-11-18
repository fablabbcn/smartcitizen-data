''' Long processing: another blueprint for the same device, and data read from the backups '''
import os
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


def test_backup_to_a_local_folder(tmp_path):
    # Backups written to a local folder are read back as those in S3
    from types import SimpleNamespace
    from scdata.device.device import Device

    device = SimpleNamespace(id=1, handler=SimpleNamespace(json=SimpleNamespace(model_dump_json=lambda: '{"id": 1}')))
    device._backup_to_folder = lambda *args: Device._backup_to_folder(device, *args)

    def backup(start, periods, mode):
        index = pd.date_range(start, periods=periods, freq='1min', tz='UTC', name='TIME')
        device.data = pd.DataFrame({'TEMP': 20.0}, index=index)
        return Device.backup_to_storage(device, mode=mode, root=str(tmp_path))

    assert backup('2026-01-01', 60, 'overwrite')
    assert backup('2026-01-01 01:00', 60, 'append')
    data = read_storage(str(tmp_path / 'devices' / '1' / 'data'))
    assert len(data) == 120 and data.index[0] == pd.Timestamp('2026-01-01', tz='UTC')
    assert (tmp_path / 'devices' / '1' / 'metadata.json').read_text() == '{"id": 1}'

    # Overwrite starts again
    assert backup('2026-02-01', 10, 'overwrite')
    assert len(read_storage(str(tmp_path / 'devices' / '1' / 'data'))) == 10
    # qc data is only backed up to S3
    assert not Device.backup_to_storage(device, root=str(tmp_path), add_qc_data=True)


def test_read_storage_with_channels_of_later_backups(tmp_path):
    # A channel added after the first backup is read too, empty before
    for number, columns in enumerate([{'TEMP': 1.0}, {'TEMP': 2.0, 'NEW': 3.0}]):
        index = pd.date_range(f'2026-01-0{number + 1}', periods=3, freq='1min', tz='UTC')
        pd.DataFrame({**columns, 'TIME': index}).to_parquet(tmp_path / f'part-{number}.parquet', index=False)

    data = read_storage(str(tmp_path))

    assert sorted(data.columns) == ['NEW', 'TEMP'] and len(data) == 6
    assert data.loc['2026-01-01', 'NEW'].isna().all()
    assert (data.loc['2026-01-02', 'NEW'] == 3.0).all()
    assert list(read_storage(str(tmp_path), channels=['NEW']).columns) == ['NEW']


def local_device(start, periods):
    from types import SimpleNamespace
    from scdata.device.device import Device

    device = SimpleNamespace(id=1, handler=SimpleNamespace(json=SimpleNamespace(model_dump_json=lambda: '{"id": 1}')))
    device._backup_to_folder = lambda *args: Device._backup_to_folder(device, *args)
    device.data = pd.DataFrame({'TEMP': 20.0}, index=pd.date_range(start, periods=periods, freq='1min', tz='UTC',
                                                                   name='TIME'))
    return device


@pytest.mark.parametrize('mode', ['overwrite', 'append'])
def test_failed_backup_to_a_local_folder_keeps_the_previous_one(tmp_path, monkeypatch, mode):
    from scdata.device.device import Device

    assert Device.backup_to_storage(local_device('2026-01-01', 60), root=str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError('No space left on device')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fail)
    with pytest.raises(OSError):
        Device.backup_to_storage(local_device('2026-02-01', 10), mode=mode, root=str(tmp_path))

    # The previous backup, without partial files or what was written aside
    assert len(read_storage(str(tmp_path / 'devices' / '1' / 'data'))) == 60
    assert sorted(path.name for path in (tmp_path / 'devices' / '1').iterdir()) == ['data', 'metadata.json']


@pytest.mark.parametrize('mode', ['overwrite', 'append'])
@pytest.mark.parametrize('failing', ['model_dump_json', 'write', 'replace'])
def test_failed_metadata_of_a_local_backup_keeps_the_previous_one(tmp_path, monkeypatch, failing, mode):
    import builtins
    from scdata.device.device import Device

    assert Device.backup_to_storage(local_device('2026-01-01', 60), root=str(tmp_path))

    device = local_device('2026-02-01', 10)
    device.handler.json.model_dump_json = lambda: '{"id": 2}'
    if failing == 'model_dump_json':
        def fail():
            raise ValueError('Cannot serialize')
        device.handler.json.model_dump_json = fail
    elif failing == 'write':
        # The disk fills while writing it
        class Full:
            def __init__(self, file): self.file = file
            def __enter__(self): return self
            def __exit__(self, *args): self.file.close()
            def write(self, text): raise OSError('No space left on device')
        real_open = builtins.open
        monkeypatch.setattr(builtins, 'open', lambda path, *args, **kwargs: Full(real_open(path, *args, **kwargs))
                            if str(path).endswith('.json') else real_open(path, *args, **kwargs))
    else:
        # The data is in place, but metadata.json cannot be replaced
        real_replace = os.replace
        def replace(source, destination):
            if str(destination).endswith('metadata.json'):
                raise OSError('Permission denied')
            real_replace(source, destination)
        monkeypatch.setattr(os, 'replace', replace)
    with pytest.raises((ValueError, OSError)):
        Device.backup_to_storage(device, mode=mode, root=str(tmp_path))
    monkeypatch.undo()

    # The previous backup: its data and its metadata
    assert len(read_storage(str(tmp_path / 'devices' / '1' / 'data'))) == 60
    assert (tmp_path / 'devices' / '1' / 'metadata.json').read_text() == '{"id": 1}'
    assert sorted(path.name for path in (tmp_path / 'devices' / '1').iterdir()) == ['data', 'metadata.json']

def test_backup_to_s3_only_goes_to_the_bucket(monkeypatch):
    from scdata.device.device import Device

    monkeypatch.setenv('S3_DATA_BUCKET', 'bucket')
    assert not Device.backup_to_storage(local_device('2026-01-01', 10), root='s3://another/devices')


def test_metadata_of_a_local_backup_is_utf8(tmp_path, monkeypatch):
    # Whatever the encoding of the system, as in S3
    import builtins
    from scdata.device.device import Device

    real_open = builtins.open
    def ascii_open(path, mode='r', *args, **kwargs):
        if 'b' not in mode:
            kwargs.setdefault('encoding', 'ascii')
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', ascii_open)
    device = local_device('2026-01-01', 10)
    device.handler.json.model_dump_json = lambda: '{"name": "Plaça del Sol"}'

    assert Device.backup_to_storage(device, root=str(tmp_path))
    assert (tmp_path / 'devices' / '1' / 'metadata.json').read_text(encoding='utf-8') == '{"name": "Plaça del Sol"}'
