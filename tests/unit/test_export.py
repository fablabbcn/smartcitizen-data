from os.path import exists, join

import pytest
from pandas import DataFrame, read_csv

from conftest import SHT31_TEMP


def processed_device(make_device, index, exports=None):
    blueprint = {'channels': [], 'exports': exports or []}
    device = make_device(blueprint, [SHT31_TEMP])
    device.data = DataFrame({'TEMP': 20.0, 'HUM': 50.0}, index=index)
    device.loaded = True
    return device


def test_export_without_exports(make_device, index, tmp_path):
    device = processed_device(make_device, index)

    assert device.export(str(tmp_path)) is True
    assert exists(join(tmp_path, '1.csv'))


def test_export_with_exports(make_device, index, tmp_path):
    device = processed_device(make_device, index, [{'name': 'all'}, {'name': 'clean', 'columns': ['TEMP']}])

    assert device.export(str(tmp_path)) is True
    assert exists(join(tmp_path, '1_data_all.csv'))
    assert exists(join(tmp_path, '1_data_clean.csv'))
    assert not exists(join(tmp_path, '1.csv'))


def test_export_ignoring_exports(make_device, index, tmp_path):
    device = processed_device(make_device, index, [{'name': 'clean', 'columns': ['TEMP']}])

    assert device.export(str(tmp_path), use_exports=False) is True
    assert exists(join(tmp_path, '1.csv'))


def test_export_existing_file_without_overwrite(make_device, index, tmp_path):
    device = processed_device(make_device, index)
    device.export(str(tmp_path))

    assert device.export(str(tmp_path)) is False
    assert device.export(str(tmp_path), forced_overwrite=True) is True


def test_export_empty_data(make_device, tmp_path):
    device = make_device({'channels': []}, [SHT31_TEMP])

    assert device.export(str(tmp_path)) is False


def test_export_selected_and_renamed_columns(make_device, index, tmp_path):
    device = processed_device(make_device, index)
    device.data = device.data.assign(SO2=1.0, SO2_WE=2.0, SO2_ALS=3.0, SO2_custom=4.0, SO2_custom_ugm3=5.0)

    assert device.export(str(tmp_path), exclude=['SO2', 'SO2_ALS*'], rename={'SO2_custom*': 'SO2*'}) is True

    exported = read_csv(join(tmp_path, '1.csv'), index_col=0)
    assert list(exported.columns) == ['TEMP', 'HUM', 'SO2_WE', 'SO2', 'SO2_ugm3']
    assert exported['SO2'].iloc[0] == 4.0 and exported['SO2_ugm3'].iloc[0] == 5.0


def test_export_only_some_columns(make_device, index, tmp_path):
    device = processed_device(make_device, index)

    assert device.export(str(tmp_path), columns=['T*']) is True
    assert list(read_csv(join(tmp_path, '1.csv'), index_col=0).columns) == ['TEMP']


def test_export_fails_with_columns_renamed_to_the_same_name(make_device, index, tmp_path):
    device = processed_device(make_device, index)
    device.data = device.data.assign(SO2=1.0, SO2_custom=4.0)

    with pytest.raises(ValueError, match='SO2'):
        device.export(str(tmp_path), rename={'SO2_custom*': 'SO2*', 'SO2': 'SO2'})
    assert not exists(join(tmp_path, '1.csv'))
