from os.path import exists, join

from pandas import DataFrame

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
