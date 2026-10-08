from scdata.io.device_file import sdcard_concat

HEADER = 'TIME,TEMP,HUM\nUTC,C,%\nTime,Temperature,Humidity\n,55,56\n'


def write(path, name, rows):
    (path / name).write_text(HEADER + ''.join(f'{row}\n' for row in rows))


def test_concat_files(tmp_path):
    write(tmp_path, '26-01-01.CSV', ['2026-01-01T10:00:00Z,20.0,50.0', '2026-01-01T10:01:00Z,21.0,51.0'])
    write(tmp_path, '26-01-02.CSV', ['2026-01-02T10:00:00Z,22.0,52.0'])

    concat = sdcard_concat(str(tmp_path), output='', timezone='UTC')

    assert len(concat) == 3
    assert concat['TEMP'].tolist() == [20.0, 21.0, 22.0]


def test_concat_writes_output(tmp_path):
    write(tmp_path, '26-01-01.CSV', ['2026-01-01T10:00:00Z,20.0,50.0'])

    sdcard_concat(str(tmp_path), output='CONCAT.CSV', timezone='UTC')

    assert (tmp_path / 'CONCAT.CSV').exists()


def test_concat_skips_files_without_rows(tmp_path):
    write(tmp_path, '26-01-01.CSV', ['2026-01-01T10:00:00Z,20.0,50.0'])
    write(tmp_path, '26-01-02.CSV', [])

    concat = sdcard_concat(str(tmp_path), output='', timezone='UTC')

    assert len(concat) == 1


def test_concat_combines_rows_duplicated_after_localisation(tmp_path):
    # Same instant written with different offsets, with values in different columns
    write(tmp_path, '26-01-01.CSV', ['2026-01-01T10:00:00Z,20.0,'])
    write(tmp_path, '26-01-02.CSV', ['2026-01-01T11:00:00+01:00,,50.0'])

    concat = sdcard_concat(str(tmp_path), output='', timezone='UTC')

    assert len(concat) == 1
    assert concat.iloc[0].to_dict() == {'TEMP': 20.0, 'HUM': 50.0}


def test_concat_renames_with_the_names(tmp_path, monkeypatch):
    from scdata._config import config
    from scdata.models import Name

    names = [Name(id=55, name='TEMP', description='', unit='C'),
             Name(id=89, name='PMS5003_PM_1', description='', unit='ug/m3'),
             Name(id=89, name='SECOND_NAME', description='', unit='ug/m3'),
             Name(id=0, name='NO_ID', description='', unit='')]
    monkeypatch.setitem(config.names, 'SCDevice', names)
    (tmp_path / '26-01-01.CSV').write_text('TIME,TEMP,PM_1,EXTRA\nUTC,C,ug/m3,x\nTime,Temperature,PM 1,Extra\n,55,89,0\n'
                                           '2026-01-01T10:00:00Z,20.0,3.0,1.0\n')

    concat = sdcard_concat(str(tmp_path), output='CONCAT.CSV', timezone='UTC', rename=True)

    assert set(concat.columns) == {'TEMP', 'PMS5003_PM_1', 'EXTRA'}
    # The header rows of the output follow the renamed columns
    header = [row.split(',')[1:] for row in (tmp_path / 'CONCAT.CSV').read_text().splitlines()[:4]]
    assert dict(zip(header[0], header[3])) == {'TEMP': '55', 'PMS5003_PM_1': '89', 'EXTRA': '0'}


def test_concat_blueprint_is_a_deprecated_rename(tmp_path, monkeypatch):
    from scdata._config import config
    from scdata.models import Name

    monkeypatch.setitem(config.names, 'SCDevice', [Name(id=56, name='HUMIDITY', description='', unit='%')])
    write(tmp_path, '26-01-01.CSV', ['2026-01-01T10:00:00Z,20.0,50.0'])

    concat = sdcard_concat(str(tmp_path), output='', timezone='UTC', blueprint='sc_air')

    assert set(concat.columns) == {'TEMP', 'HUMIDITY'}
