''' Health checks: gaps with their settings, and the summary stored in device.health '''
import numpy as np
import pandas as pd
import pytest

from scdata.device.check import find_gaps, find_implausible_values, find_insensitive_electrodes
from scdata.device.check.gaps import gap_intervals
from scdata.device.check.summary import flagged_runs, summarise_check


def minutes(*values):
    return pd.DatetimeIndex([pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(minutes=value) for value in values])


def series(times, values=None):
    index = minutes(*times)
    return pd.Series(values if values is not None else 1.0, index=index, dtype=float)


# Gaps

def test_gap_is_longer_than_the_gap_size():
    readings = series([0, 1, 2, 3, 10, 11, 12])

    assert gap_intervals(readings, gap_size_minutes=5, frequency_minutes=1) == [(minutes(3)[0], minutes(10)[0])]
    assert gap_intervals(readings, gap_size_minutes=10, frequency_minutes=1) == []


def test_frequency_longer_than_gap_size_is_not_a_gap():
    # A sensor every 5 minutes, default gap size of 1 minute
    readings = series([0, 5, 10, 15])

    assert gap_intervals(readings, gap_size_minutes=1, frequency_minutes=5) == []
    assert gap_intervals(readings, gap_size_minutes=1, frequency_minutes=1) != []


def test_missing_readings_at_the_ends_are_gaps():
    readings = series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [np.nan] * 6 + [1.0] * 5)

    assert gap_intervals(readings, 5, 1) == [(minutes(0)[0], minutes(6)[0])]


def test_column_without_readings_is_one_gap():
    readings = series([0, 10], [np.nan, np.nan])

    assert gap_intervals(readings, 5, 1) == [(minutes(0)[0], minutes(10)[0])]


def test_find_gaps_uses_the_settings_of_each_column():
    index = minutes(*range(0, 21))
    data = pd.DataFrame({'TEMP': 1.0, 'PM': np.nan}, index=index)
    data.loc[index[::5], 'PM'] = 2.0                  # every 5 minutes
    data.loc[index[5:12], 'TEMP'] = np.nan            # 7 minutes without TEMP

    result = find_gaps(data, default_gap_size_minutes=5,
                       frequencies=[{'columns': ['PM'], 'frequency_minutes': 5}])

    assert result.data['__TEMP'].sum() == 7
    assert result.intervals['__TEMP'] == [(index[4], index[12])]
    assert not result.data['__PM'].any() and result.intervals['__PM'] == []

    result = find_gaps(data, gap_sizes=[{'columns': ['TEMP'], 'gap_size_minutes': 10}],
                       frequencies=[{'columns': ['PM'], 'frequency_minutes': 5}])

    assert result.intervals['__TEMP'] == []


# Summary

@pytest.mark.parametrize('missing, dtype', [(np.nan, None), (None, object), (pd.NA, 'boolean')])
def test_flagged_runs(missing, dtype):
    index = minutes(0, 1, 2, 3, 4)

    # Missing values are not flagged
    assert flagged_runs(pd.Series([True, True, False, True, missing], index=index, dtype=dtype)) == [
        (index[0], index[1]), (index[3], index[3])]


def test_summary_of_a_value_check():
    index = minutes(0, 1, 2, 3)
    data = pd.DataFrame({'NOISE_A': [50, 120, 130, np.nan]}, index=index)

    summary = summarise_check(find_implausible_values(
        data, implausible_values=[{'column': 'NOISE_A', 'limits': [20, 99]}]), data)

    assert summary == {'NOISE_A': {'flagged': 2, 'checked': 3, 'ratio': 0.6667, 'more_intervals': 0,
                                   'intervals': [[index[1].isoformat(), index[2].isoformat()]]}}


def test_summary_of_gaps_is_in_time():
    # A full outage leaves no rows: only the interval shows it
    index = minutes(0, 1, 2, 3, 60)
    data = pd.DataFrame({'TEMP': 1.0}, index=index)

    summary = summarise_check(find_gaps(data), data)['TEMP']

    assert summary['flagged'] == 0
    assert (summary['minutes'], summary['ratio']) == (57.0, 0.95)
    assert summary['intervals'] == [[index[3].isoformat(), index[4].isoformat()]]


# Device

CHECKS = [
    {'name': 'GAPS', 'function': 'find_gaps', 'kwargs': {'default_gap_size_minutes': 5}},
    {'name': 'IMPLAUSIBLE', 'function': 'find_implausible_values',
     'kwargs': {'implausible_values': [{'column': 'ADC_48_3', 'limits': [0, 1]}]}},
    {'name': 'BROKEN', 'function': 'find_implausible_values', 'kwargs': {'implausible_values': [{'column': 'ADC_48_3'}]}},
]


@pytest.fixture
def checked_device(make_device):
    index = minutes(*range(0, 10), *range(30, 40))
    readings = pd.DataFrame({'ADC_48_3': 0.5}, index=index)
    readings.iloc[-1] = 2.0
    device = make_device({'channels': [], 'checks': CHECKS}, [{'id': 139, 'name': 'ADC_48_3', 'description': 'ADC',
                                                                  'unit': 'V'}])
    device.data = readings
    device.loaded = True
    return device


def test_health_checks_fill_device_health(checked_device):
    checked_device.health_checks()

    health = checked_device.health
    assert (health['rows'], health['start'], health['end']) == (20, minutes(0)[0].isoformat(), minutes(39)[0].isoformat())
    gaps, implausible, broken = health['checks']
    assert (gaps['name'], gaps['status']) == ('GAPS', 'ok')
    assert gaps['columns']['ADC_48_3']['minutes'] == 21.0
    assert implausible['columns']['ADC_48_3']['flagged'] == 1
    assert broken['status'].startswith('error: KeyError')
    assert broken['columns'] == {}


# Insensitive electrodes

def electrode_pair(peaks):
    ''' Six days every minute: the AE and the WE follow the temperature, and the WE has the gas peaks, in V '''
    index = pd.date_range('2026-01-01', periods=6 * 1440, freq='1min', tz='UTC')
    hours = np.arange(len(index)) / 60
    rng = np.random.default_rng(0)
    temperature = 0.002 * np.sin(2 * np.pi * hours / 24)
    gas = peaks * 0.01 * (np.sin(2 * np.pi * hours / 7) > 0.9)
    return pd.DataFrame({'SO2_AE': 0.35 + temperature + rng.normal(0, 1e-4, len(index)),
                         'SO2_WE': 0.33 + 2 * temperature + gas + rng.normal(0, 1e-4, len(index))}, index=index)


@pytest.mark.parametrize('peaks, flagged', [(False, True), (True, False)])
def test_a_working_electrode_without_gas_response_is_flagged(peaks, flagged):
    data = electrode_pair(peaks)

    result = find_insensitive_electrodes(data, electrodes=[{'channel': 'SO2', 'we': 'SO2_WE', 'ae': 'SO2_AE'}])

    assert result.status_code.name == 'SUCCESS'
    assert result.data['__SO2'].all() == flagged and result.data['__SO2'].any() == flagged


def test_electrodes_that_are_not_in_the_data_are_skipped():
    result = find_insensitive_electrodes(electrode_pair(False), electrodes=[{'channel': 'NO2', 'we': 'NO2_WE', 'ae': 'NO2_AE'}])

    assert result.data.empty and list(result.data.columns) == []


def test_only_readings_with_the_channel_are_flagged():
    data = electrode_pair(False)
    data['SO2'] = 1.0
    data.iloc[:360, data.columns.get_loc('SO2')] = np.nan

    flags = find_insensitive_electrodes(data, electrodes=[{'channel': 'SO2', 'we': 'SO2_WE', 'ae': 'SO2_AE'}]).data['__SO2']

    assert not flags.iloc[:360].any() and flags.iloc[360:].all()
