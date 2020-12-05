from re import search
from scdata.utils.out import std_out
from scdata._config import config

def get_units_convf(sensor, from_units):
    """
    Returns a factor which will be multiplied to sensor. It accounts for unit
    convertion based on the desired units in the config._channel_lut for each sensor.
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

    for channel in config._channel_lut.keys():
        if not (search(channel, sensor)): continue
        # Molecular weight in case of pollutants
        for pollutant in config._molecular_weights.keys():
            if search(channel, pollutant): 
                molecular_weight = config._molecular_weights[pollutant]
                break
            else: molecular_weight = 1
        
        # Check if channel is in look-up table
        if config._channel_lut[channel] != from_units: 
            std_out(f"Converting units for {sensor}. From {from_units} to {config._channel_lut[channel]}")
            for unit in config._unit_convertion_lut:
                # Get units
                if unit[0] == from_units and unit[1] == config._channel_lut[channel]: 
                    factor = unit[2]
                    requires_conc = unit[3]
                    break
                elif unit[1] == from_units and unit[0] == config._channel_lut[channel]: 
                    factor = 1/unit[2]
                    requires_conc = unit[3]
                    break
            if requires_conc: rfactor = factor/molecular_weight
            else: rfactor = factor
            std_out(f"Factor: {rfactor}")
        else: 
            std_out(f"No units conversion needed for {sensor}")
            rfactor = 1
        if rfactor != 1: break
    
    return rfactor