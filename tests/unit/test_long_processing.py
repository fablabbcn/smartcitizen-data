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


def test_load_from_storage(make_device, backup):
    device = make_device(SHORT, ADC)

    assert device.load_from_storage(root=str(backup), min_date='2026-01-11 12:00')

    # The second file ends at 2026-01-11 23:58
    assert device.loaded and len(device.data) == 719 and set(device.data.columns) == {'TEMP', 'HUM'}
    assert not device.load_from_storage(root=str(backup / 'nowhere'))


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
