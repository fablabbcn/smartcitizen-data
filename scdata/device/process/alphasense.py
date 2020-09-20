from scdata.utils import std_out, get_units_convf, find_dates
from scdata._config import config
from scdata.device.process import baseline_calc
from scipy.stats.stats import linregress
import matplotlib.pyplot as plt
from pandas import date_range, DataFrame, Series

def basic_4electrode_alg(dataframe, **kwargs):
    """
    Calculates pollutant concentration based on 4 electrode sensor readings (mV)
    and calibration ID. It adds a configurable background concentration.
    Parameters
    ----------
        working: string
            Name of working electrode found in dataframe
        auxiliary: string
            Name of auxiliary electrode found in dataframe
        id: int 
            Sensor ID
        pollutant: string
            Pollutant name. Must be included in the corresponding LUTs for unit convertion and additional parameters:
            MOLECULAR_WEIGHTS, config._background_conc, CHANNEL_LUT
        hardware: alphadelta or isb
    Returns
    -------
        calculation of pollutant based on: 6.36 * sensitivity(working - zero_working)/(auxiliary - zero_auxiliary)
    """
    flag_error = False
    if 'working' not in kwargs: flag_error = True
    if 'auxiliary' not in kwargs: flag_error = True
    if 'id' not in kwargs: flag_error = True
    if 'pollutant' not in kwargs: flag_error = True

    if flag_error: 
        std_out('Problem with input data', 'ERROR')
        return None

    # Get Sensor data
    if kwargs['id'] not in config.calibrations: 
        std_out(f"Sensor {kwargs['id']} not in calibration data", 'ERROR')
        return None

    we_sensitivity_na_ppb = config.calibrations[kwargs['id']]['we_sensitivity_na_ppb']
    we_cross_sensitivity_no2_na_ppb = config.calibrations[kwargs['id']]['we_cross_sensitivity_no2_na_ppb']
    sensor_type = config.calibrations[kwargs['id']]['sensor_type']
    # TODO - implement different algorithms
    nWA = config.calibrations[kwargs['id']]['we_sensor_zero_mv']/config.calibrations[kwargs['id']]['ae_sensor_zero_mv']

    if sensor_type != kwargs['pollutant']: 
        std_out(f"Sensor {kwargs['id']} doesn't coincide with calibration data", 'ERROR')
        return None

    # This is always in ppm since the calibration data is in signal/ppm
    if kwargs['hardware'] == 'alphadelta': current_factor = config._alphadelta_pcb
    elif kwargs['hardware'] == 'isb': current_factor = 1 #TODO make it so we talk in mV
    else: 
        std_out(f"Measurement hardware {kwargs['hardware']} not supported", 'ERROR')
        return None

    result = current_factor*(dataframe[kwargs['working']] - nWA*dataframe[kwargs['auxiliary']])/abs(we_sensitivity_na_ppb)

    # Convert units
    result *= get_units_convf(kwargs['pollutant'], from_units = 'ppm')
    # Add Background concentration
    result += config._background_conc[kwargs['pollutant']]

    return result

def baseline_4electrode_alg(dataframe, **kwargs):
    """
    Calculates pollutant concentration based on 4 electrode sensor readings (mV), but using
    one of the metrics (baseline) as a baseline of the others. It uses the baseline correction algorithm
    explained here: 
    https://docs.smartcitizen.me/Components/sensors/Electrochemical%20Sensors/#baseline-correction-based-on-temperature
    and the calibration ID. It adds a configurable background concentration.
    Parameters
    ----------
        target: string
            Name of working electrode found in dataframe
        baseline: string
            Name of auxiliary electrode found in dataframe
        id: int 
            Sensor ID
        pollutant: string
            Pollutant name. Must be included in the corresponding LUTs for unit convertion and additional parameters:
            MOLECULAR_WEIGHTS, config._background_conc, CHANNEL_LUT
        regression_type: 'string'
            'best'
            Use a 'linear' or 'exponential' regression for the calculation of the baseline
        period: pd.offset_alias or 'full'
            1D
            The period at which the baseline is calculated. If full, the whole index will be used.
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        store_baseline: boolean
            True
            Whether or not to store the baseline in the dataframe
        resample: str
            '1Min'
            Resample frequency for the target dataframe         
        pcb_factor: int
            config._alphadelta_pcb (6.36)
            Factor converting mV to nA due to the board configuration

    Returns
    -------
        calculation of pollutant based on: pcb_factor*sensitivity(working - zero_working)/(auxiliary - zero_auxiliary)
    """

    result = Series()
    baseline = Series()

    flag_error = False
    if 'target' not in kwargs: flag_error = True
    if 'baseline' not in kwargs: flag_error = True
    if 'id' not in kwargs: flag_error = True
    if 'pollutant' not in kwargs: flag_error = True
    
    if 'regression_type' in kwargs: 
        if kwargs['regression_type'] not in ['best', 'exponential', 'linear']: flag_error = True
        else: reg_type = kwargs['regression_type']
    else: reg_type = 'best'
    
    if 'period' in kwargs: 
        if kwargs['period'] not in ['best', 'exponential', 'linear']: flag_error = True
        else: period = kwargs['period']
    else: period = '1D'

    if 'store_baseline' in kwargs: store_baseline = kwargs['store_baseline']
    else: store_baseline = True

    if 'resample' in kwargs: resample = kwargs['resample']
    else: resample = '1Min'    

    if 'pcb_factor' in kwargs: pcb_factor = kwargs['pcb_factor']
    else: pcb_factor = config._alphadelta_pcb
    
    if 'baseline_type' in kwargs: baseline_type = kwargs['baseline_type']
    else: baseline_type = 'deltas'

    if 'deltas' in kwargs: deltas = kwargs['deltas']
    else: deltas = config._baseline_deltas
    
    if flag_error: 
        std_out('Problem with input data', 'ERROR')
        return None

    min_date, max_date, _ = find_dates(dataframe)
    pdates = date_range(start = min_date, end = max_date, freq = period)

    for pos in range(0, len(pdates)-1):
        chunk = dataframe.loc[pdates[pos]:pdates[pos+1], [kwargs['target'], kwargs['baseline']]]
        bchunk = baseline_calc(chunk, reg_type = reg_type, resample = resample, baseline_type = baseline_type, deltas = deltas)
        if bchunk is None: continue
        baseline = baseline.combine_first(bchunk)

    if kwargs['pollutant'] not in config.convolved_metrics:

        sensitivity_1 = config.calibrations.loc[kwargs['id'],'sensitivity_1']
        sensitivity_2 = config.calibrations.loc[kwargs['id'],'sensitivity_2']
        target_1 = config.calibrations.loc[kwargs['id'],'target_1']
        target_2 = config.calibrations.loc[kwargs['id'],'target_2']
        nWA = config.calibrations.loc[kwargs['id'],'w_zero_current']/config.calibrations.loc[kwargs['id'],'aux_zero_current']

        if target_1 != kwargs['pollutant']: 
            std_out(f"Sensor {kwargs['id']} doesn't coincide with calibration data", 'ERROR')
            return None
        
        result = pcb_factor*(dataframe[kwargs['target']] - baseline)/abs(sensitivity_1)

        # Convert units
        result *= get_units_convf(kwargs['pollutant'], from_units = 'ppm')
        # Add Background concentration
        result += config._background_conc[kwargs['pollutant']]
    
    else:
        # Calculate non convolved part
        result = dataframe[kwargs['target']] - baseline

    # Make use of DataFrame inmutable properties to store in it the baseline
    if store_baseline:
        dataframe[kwargs['target']+'_BASELINE'] = baseline
    
    return result

def deconvolution(dataframe, **kwargs):
    """
    Calculates pollutant concentration for convolved metrics, such as NO2+O3.
    Needs convolved metric, and target pollutant sensitivities
    Parameters
    ----------
        source: string
            Name of convolved metric containing both pollutants (such as NO2+O3)
        base: string
            Name of one of the already deconvolved pollutants (for instance NO2)
        id: int 
            Sensor ID
        pollutant: string
            Pollutant name. Must be included in the corresponding LUTs for unit convertion and additional parameters:
            MOLECULAR_WEIGHTS, config._background_conc, CHANNEL_LUT
    Returns
    -------
        calculation of pollutant based on: 6.36 * sensitivity(working - zero_working)/(auxiliary - zero_auxiliary)
    """

    result = Series()
    baseline = Series()

    flag_error = False
    if 'source' not in kwargs: flag_error = True
    if 'base' not in kwargs: flag_error = True
    if 'id' not in kwargs: flag_error = True
    if 'pollutant' not in kwargs: flag_error = True

    if flag_error: 
        std_out('Problem with input data', 'ERROR')
        return None

    sensitivity_1 = config.calibrations.loc[kwargs['id'],'sensitivity_1']
    sensitivity_2 = config.calibrations.loc[kwargs['id'],'sensitivity_2']
    target_1 = config.calibrations.loc[kwargs['id'],'target_1']
    target_2 = config.calibrations.loc[kwargs['id'],'target_2']
    nWA = config.calibrations.loc[kwargs['id'],'w_zero_current']/config.calibrations.loc[kwargs['id'],'aux_zero_current']

    if target_1 != kwargs['pollutant']: 
        std_out(f"Sensor {kwargs['id']} doesn't coincide with calibration data", 'ERROR')
        return None

    factor_unit_1 = get_units_convf(kwargs['pollutant'], from_units = 'ppm')
    factor_unit_2 = get_units_convf(kwargs['base'], from_units = 'ppm')

    result = factor_unit_1*(config._alphadelta_pcb*dataframe[kwargs['source']] - dataframe[kwargs['base']]/factor_unit_2*abs(sensitivity_2))/abs(sensitivity_1)
    
    # Add Background concentration
    result += config._background_conc[kwargs['pollutant']]
    
    return result