''' Units of the API converted to the units of the channels '''
from scdata.tools.units import get_units_convf


def test_same_units():
    assert get_units_convf('TEMP', from_units='degC') == 1


def test_known_conversions():
    assert get_units_convf('TEMP', from_units='ºC') == 1
    assert get_units_convf('NO2', from_units='ppm') == 1000
    # ug/m3 to ppb, with the molecular weight of CO
    assert get_units_convf('CO', from_units='ug/m3') == 24.45 / 28


def test_unknown_conversion_is_not_converted():
    # As in names/SCDevice.json
    assert get_units_convf('TEMP', from_units='C') == 1
