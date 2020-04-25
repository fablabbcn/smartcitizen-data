from re import search
from scdata.utils.out import std_out
from scdata._config import config

def get_units_convf(sensor, from_units):
    """
    Returns a factor which will be multiplied to sensor. It accounts for unit
    convertion based on the desired units in the config.channel_lut for each sensor.
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

    rfactor = None
    for channel in config.channel_lut.keys():
        if not (search(channel, sensor)): continue
        # Molecular weight in case of pollutants
        for pollutant in config.molecular_weights.keys(): 
            if search(channel, pollutant): 
                molecular_weight = config.molecular_weights[pollutant]
                break
            else: molecular_weight = 1
        
        # Check if channel is in look-up table
        if config.channel_lut[channel] != from_units: 
            std_out(f"Converting units for {sensor}. From {from_units} to {config.channel_lut[channel]}")
            for unit in config.unit_convertion_lut:
                # Get units
                if unit[0] == from_units: 
                    factor = unit[2]
                    break
                elif unit[1] == from_units: 
                    factor = 1/unit[2]
                    break
            rfactor = factor/molecular_weight
        else: 
            std_out(f"No units conversion needed for {sensor}")
            rfactor = 1
        if rfactor is not None: break
    return rfactor