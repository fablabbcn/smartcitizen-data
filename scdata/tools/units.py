from re import search
from scdata.tools.custom_logger import logger

# Molecular weights of certain pollutants for unit convertion
molecular_weights = {
    'CO':   28,
    'NO':   30,
    'NO2':  46,
    'O3':   48,
    'C6H6': 78,
    'SO2':  64,
    'H2S':  34
}

# This look-up table is comprised of channels you want always want to have with the same units and that might come from different sources
# i.e. pollutant data in various units (ppm or ug/m3) from different analysers
# The table should be used as follows:
# 'key': 'units',
# - 'key' is the channel that will lately be used in the analysis. It supports regex
# - target_unit is the unit you want this channel to be and that will be converted in case of it being found in the channels list of your source

# TODO - move to units in pypi
channel_lut = {
    "TEMP": "degC",
    "HUM": "%rh",
    "PRESS": "kPa",
    "PM_(\d|[A,B]_\d)": "ug/m3",
    "^CO2": "ppm",
    "^CO": "ppb", # Always start with CO
    "NOISE_A": "dBA",
    "NO\Z": "ppb",
    "NO2": "ppb",
    "NOX": "ppb",
    "O3": "ppb",
    "C6H6": "ppb",
    "H2S": "ppb",
    "SO2": "ppb",
    "CO2": "ppm"
}

# This table is used to convert units
# ['from_unit', 'to_unit', 'multiplicative_factor', 'requires_M']
# - 'from_unit'/'to_unit' = 'multiplicative_factor'
# - 'requires_M' = whether it
# It accepts reverse operations - you don't need to put them twice but in reverse

unit_convertion_lut = (
    ['%rh', '%', 1, False],
    ['ºC', 'degC', 1, False],
    ['ppm', 'ppb', 1000, False],
    ['mg/m3', 'ug/m3', 1000, False],
    ['mgm3', 'ugm3', 1000, False],
    ['mg/m3', 'ppm', 24.45, True],
    ['mgm3', 'ppm', 24.45, True],
    ['ug/m3', 'ppb', 24.45, True],
    ['ugm3', 'ppb', 24.45, True],
    ['mg/m3', 'ppb', 1000*24.45, True],
    ['mgm3', 'ppb', 1000*24.45, True],
    ['ug/m3', 'ppm', 1./1000*24.45, True],
    ['ugm3', 'ppm', 1./1000*24.45, True]
)

def get_units_convf(sensor, from_units):
    """
    Returns a factor which will be multiplied to sensor. It accounts for unit
    convertion based on the desired units in the channel_lut for each sensor.
    channel_converted = factor * sensor
    Parameters
    ----------
        sensor: string
            Name of the sensor channel
        from_units: string
            Units in which it currently is
    Returns
    -------
        factor (float)
        factor = unit_convertion_factor/molecular_weight
    Note:
        This would need to be changed if all pollutants were to be expresed in
        mass units, instead of ppm/b
    """

    rfactor = 1

    for channel in channel_lut.keys():
        if not (search(channel, sensor)): continue
        # Molecular weight in case of pollutants
        for pollutant in molecular_weights.keys():
            if search(channel, pollutant):
                molecular_weight = molecular_weights[pollutant]
                break
            else: molecular_weight = 1

        # Check if channel is in look-up table
        if channel_lut[channel] != from_units and from_units != "" and from_units is not None:
            logger.info(f"Converting units for {sensor}. From {from_units} to {channel_lut[channel]}")
            for unit in unit_convertion_lut:
                # Get units
                if unit[0] == from_units and unit[1] == channel_lut[channel]:
                    factor = unit[2]
                    requires_conc = unit[3]
                    break
                elif unit[1] == from_units and unit[0] == channel_lut[channel]:
                    factor = 1/unit[2]
                    requires_conc = unit[3]
                    break
            if requires_conc: rfactor = factor/molecular_weight
            else: rfactor = factor
        else:
            logger.info(f"No units conversion needed for {sensor}. Same units")
            if from_units == "":
                logger.info("Empty units in blueprint is placeholder for keep")
            rfactor = 1
        if rfactor != 1: break

    return rfactor
