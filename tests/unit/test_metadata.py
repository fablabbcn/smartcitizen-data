''' Consistency of the metadata files: blueprints, names, hardware and calibrations '''
import glob
from collections import Counter
from os.path import basename, join

import pytest
from smartcitizen_connector._config import config as connector_config
from smartcitizen_connector.models import HardwarePostprocessing

from scdata.models import Blueprint
from scdata.tools.tree import topological_sort

from conftest import ROOT, load_json

BLUEPRINTS = sorted(glob.glob(join(ROOT, 'blueprints', '*.json')))
HARDWARE = sorted(glob.glob(join(ROOT, 'hardware', '*.json')))

# Known issues in the current data. Remove items once fixed, never add new ones
# Names used in blueprints that are not in names/SCDevice.json
KNOWN_MISSING_NAMES = {'MPL_PRESS'}
# (hardware, sensor id) with an unknown Alphasense code
KNOWN_UNKNOWN_SENSOR_CODES = {
    ('SCAS210030', '163040003'), ('SCAS220051', '73002320'), ('SCAS220052', '73002016'),
    ('SCAS220053', '73002157'), ('SCAS220054', '73002644'), ('SCAS220055', '73002265'),
    ('SCAS220117', 'nan'),
}
# (hardware, sensor id) not in calibrations.json
KNOWN_MISSING_CALIBRATIONS = KNOWN_UNKNOWN_SENSOR_CODES | {
    ('SCAS210030', '164780964'), ('SCAS210030', '202760040'), ('SCAS210030', '204042163'),
    ('SCAS230006', '164210262'), ('SCAS230007', '164210264'), ('SCAS230008', '164210263'),
    ('SCAS230009', '164100628'), ('SCAS230010', '164100629'),
}


def columns_in(item):
    ''' Columns referenced in checks kwargs '''
    if isinstance(item, dict):
        for key, value in item.items():
            if key in ['column', 'columns']:
                yield from [value] if isinstance(value, str) else value
            else:
                yield from columns_in(value)
    elif isinstance(item, list):
        for value in item:
            yield from columns_in(value)


@pytest.fixture(scope='module')
def name_list():
    return [name['name'] for name in load_json('names', 'SCDevice.json')]


def test_names_are_unique(name_list):
    assert [name for name, count in Counter(name_list).items() if count > 1] == []


@pytest.mark.parametrize('path', BLUEPRINTS, ids=basename)
def test_blueprint_is_valid(path):
    blueprint = Blueprint.model_validate(load_json(path))
    channel_names = [channel.name for channel in blueprint.channels]

    assert len(channel_names) == len(set(channel_names))
    for channel in blueprint.channels:
        assert set(channel.depends_on) <= set(channel_names), channel.name
    assert topological_sort(blueprint.channels) is not None


@pytest.mark.parametrize('path', BLUEPRINTS, ids=basename)
def test_blueprint_columns_are_known(path, name_list):
    blueprint = Blueprint.model_validate(load_json(path))
    known = set(name_list) | {channel.name for channel in blueprint.channels} | KNOWN_MISSING_NAMES

    for check in blueprint.checks:
        assert set(columns_in(check.kwargs)) - known == set(), check.name
    for export in blueprint.exports:
        assert set(export.columns) - known == set(), export.name


def test_known_missing_names_are_still_missing(name_list):
    assert KNOWN_MISSING_NAMES - set(name_list) == KNOWN_MISSING_NAMES


@pytest.mark.parametrize('path', HARDWARE, ids=basename)
def test_hardware(path):
    name = basename(path)[:-5]
    hardware = HardwarePostprocessing.model_validate(load_json(path))
    blueprints = [basename(blueprint) for blueprint in BLUEPRINTS]
    calibrations = load_json('calibrations', 'calibrations.json')

    assert basename(hardware.blueprint_url) in blueprints

    for version in hardware.versions:
        assert version.from_date is None or version.to_date is None or version.from_date < version.to_date
        for slot, sensor_id in version.ids.items():
            assert slot[:2] in ['AS', 'PT'], slot
            if slot.startswith('AS') and (name, sensor_id) not in KNOWN_UNKNOWN_SENSOR_CODES:
                assert sensor_id[:3] in connector_config._as_sensor_codes, sensor_id
            if (name, sensor_id) not in KNOWN_MISSING_CALIBRATIONS:
                assert sensor_id in calibrations, sensor_id


def test_known_hardware_issues_are_still_present():
    calibrations = load_json('calibrations', 'calibrations.json')
    used = set()
    for path in HARDWARE:
        for version in HardwarePostprocessing.model_validate(load_json(path)).versions:
            used |= {(basename(path)[:-5], sensor_id) for sensor_id in version.ids.values()}

    # Remove fixed items from the known issues
    assert KNOWN_MISSING_CALIBRATIONS <= used
    assert {item for item in KNOWN_MISSING_CALIBRATIONS if item[1] in calibrations} == set()
    assert {item for item in KNOWN_UNKNOWN_SENSOR_CODES
            if item[1][:3] in connector_config._as_sensor_codes} == set()
