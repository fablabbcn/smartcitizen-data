''' Electrochemical sensors (alphasense_803_04) with the options of a blueprint or a parameter set '''
import numpy as np
import pandas as pd
import pytest

from scdata._config import config
from scdata.device.process.alphasense import alphasense_803_04
from scdata.device.process.params import as_sensor_algs, as_t_comp

# ASA4_NO2: algorithm 1. ASA4_CO: algorithm 4 as the alternative
NO2 = '212070552'
CO = '132070362'


@pytest.fixture
def electrodes():
    ''' A day every minute, without readings from 12:00 to 13:00. Concentrations around 0 '''
    index = pd.date_range('2026-01-01', periods=1440, freq='1min', tz='UTC')
    index = index[(index < '2026-01-01 12:00') | (index >= '2026-01-01 13:00')]
    rng = np.random.default_rng(0)
    return pd.DataFrame({'WE': 0.2897 + rng.normal(0, 0.002, len(index)),
                         'AE': 0.303 + rng.normal(0, 0.002, len(index)), 'T': 20.0}, index=index)


def conc(data, alphasense_id=NO2, **kwargs):
    result = alphasense_803_04(data, we='WE', ae='AE', t='T', alphasense_id=alphasense_id, **kwargs)
    assert result.status_code.name == 'SUCCESS'
    return result.data


def test_negative_concentrations(electrodes):
    assert conc(electrodes).min() < 0
    assert conc(electrodes, clip_negative_conc=True).min() == 0
    assert conc(electrodes, offset_negative_conc=True).min() == pytest.approx(0)


# Parameter sets can have the units before pandas 3 (10Min, 1H)
@pytest.mark.parametrize('rolling_frequency', ['10min', '10Min', '1H'])
def test_rolling_frequency_smooths(electrodes, rolling_frequency):
    rolled = conc(electrodes, rolling_frequency=rolling_frequency)

    assert rolled.notna().all()
    assert rolled.std() < conc(electrodes).std() / 2


def test_hours_after_a_gap_are_removed(electrodes):
    result = conc(electrodes, n_gaps=10, gap_window_removal_h=1)

    # The start counts as a gap
    for start, end in (('2026-01-01 00:00', '2026-01-01 01:00'), ('2026-01-01 13:00', '2026-01-01 14:00')):
        assert result[start:end].isna().all()
    assert result['2026-01-01 01:01':'2026-01-01 11:59'].notna().all()
    assert result['2026-01-01 14:01':].notna().all()


def test_algorithm_4(electrodes):
    calibration = {key: float(value) for key, value in config.calibrations[CO].items()}
    kpp_t = np.interp(20.0, as_t_comp, as_sensor_algs['ASA4_CO'][4][1])
    we_t = electrodes['WE'] - calibration['we_electronic_zero_mv'] / 1000
    expected = (we_t - calibration['we_sensor_zero_mv'] / 1000 - kpp_t) / (calibration['we_sensitivity_mv_ppb'] / 1000)

    pd.testing.assert_series_equal(conc(electrodes, CO, use_alternative=True), expected, check_names=False)


def test_all_columns(electrodes):
    result = conc(electrodes, return_all_cols=True)

    assert list(result.columns) == ['803_we_clean', '803_ae_clean', '803_we_t', '803_ae_t', '803_we_c', 'conc']
