''' Tests of alphasense_803_04 with the options available in blueprints and parameter sets '''
import numpy as np
import pandas as pd
import pytest

from scdata._config import config
from scdata.device.process.alphasense import alphasense_803_04, alphasense_als, first_order
from scdata.device.process.params import as_sensor_algs, as_t_comp

# ASA4_NO2 uses algorithm 1, and ASA4_CO uses algorithm 4 as its alternative
NO2 = '212070552'
CO = '132070362'


@pytest.fixture
def electrodes():
    ''' One day of readings every minute, without readings from 12:00 to 13:00, and concentrations close to zero '''
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


# Parameter sets may contain frequency units that pandas 3 no longer accepts (10Min, 1H)
@pytest.mark.parametrize('rolling_frequency', ['10min', '10Min', '1H'])
def test_rolling_frequency_smooths(electrodes, rolling_frequency):
    rolled = conc(electrodes, rolling_frequency=rolling_frequency)

    assert rolled.notna().all()
    assert rolled.std() < conc(electrodes).std() / 2


def test_hours_after_a_gap_are_removed(electrodes):
    result = conc(electrodes, n_gaps=10, gap_window_removal_h=1)

    # The beginning of the data is treated as a gap
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


def test_baseline_of_a_period_shorter_than_the_hours_removed_is_an_error(electrodes):
    # The first hours after the start are removed: with only those, there is nothing to fit
    result = alphasense_als(electrodes.iloc[:120], we='WE', ae='AE', t='T', alphasense_id=NO2,
                            n_gaps=10, gap_window_removal_h=6, lam=100, p=1e-8)

    assert result.status_code.name == 'ERROR_MISSING_INPUTS'


# A WE that follows the temperature later than the AE

@pytest.fixture
def lagging():
    ''' Four days every minute, without gas: the AE follows the temperature, the WE four hours later, and humidity '''
    index = pd.date_range('2026-07-01', periods=4 * 1440, freq='1min', tz='UTC')
    hours = np.arange(len(index)) / 60
    rng = np.random.default_rng(0)
    T = pd.Series(20 + 6 * np.sin(2 * np.pi * hours / 24), index=index)
    hum = pd.Series(60 - 15 * np.sin(2 * np.pi * (hours - 2) / 24), index=index)
    return pd.DataFrame({'AE': 0.30 + 0.002 * T + rng.normal(0, 1e-5, len(index)),
                         'WE': 0.29 + 0.002 * first_order(T, 240) + 0.0001 * hum + rng.normal(0, 1e-5, len(index)),
                         'T': T, 'HUM': hum}, index=index)


def als(data, **kwargs):
    result = alphasense_als(data, we='WE', ae='AE', t='T', alphasense_id=NO2, lam=100, p=1e-8, n_gaps=10,
                            gap_window_removal_h=6, **kwargs)
    assert result.status_code.name == 'SUCCESS'
    # After the first day, when the delay has its history
    return result.data[result.data.index >= result.data.index[0] + pd.Timedelta(days=1)]


def test_first_order_reaches_two_thirds_in_its_time_constant():
    step = pd.Series(0.0, index=pd.date_range('2026-01-01', periods=1200, freq='1min'))
    step.iloc[600:] = 1.0

    delayed = first_order(step, 60)

    assert delayed.iloc[599 + 60] == pytest.approx(1 - np.exp(-1), abs=0.02)
    assert first_order(step, 0) is step


def test_without_delay_the_baseline_is_as_before(lagging):
    assert als(lagging, ae_tau_minutes=0).equals(als(lagging))


def test_a_delayed_ae_removes_the_lag_of_the_we(lagging):
    without = als(lagging, regressors=None)
    delayed = als(lagging, ae_tau_minutes='auto')

    # No gas: what is left is the temperature that the AE does not explain
    assert delayed.std() < without.std() / 2


def test_regressors_are_fitted_with_the_ae(lagging):
    assert als(lagging, ae_tau_minutes='auto', regressors=['HUM']).std() < als(lagging, ae_tau_minutes='auto').std()


def test_regressors_not_in_the_data_are_an_error(lagging):
    result = alphasense_als(lagging, we='WE', ae='AE', t='T', alphasense_id=NO2, regressors=['PRESS'])

    assert result.status_code.name == 'ERROR_MISSING_INPUTS'
