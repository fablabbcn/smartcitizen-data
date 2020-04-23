### ---------------------------------------
### ---------------IMPORTS-----------------
### ---------------------------------------

from traceback import print_exc
from termcolor import colored
from re import search, sub

from os import pardir, makedirs
from os.path import join, abspath, dirname, exists

import pandas as pd
import yaml 
import json

from urllib.request import urlopen
import requests

from geopy import distance
from src.config import Config
config = Config()

### ---------------------------------------
### ---------------LOGGING-----------------
### ---------------------------------------

def std_out(msg, mtype = None):
    out_level = config.out_level
    # Output levels:
    # 'QUIET': nothing, 
    # 'NORMAL': warn, err
    # 'DEBUG': info, warn, err, success

    if config.out_level == 'QUIET': priority = 0
    if config.out_level == 'NORMAL': priority = 1
    elif config.out_level == 'DEBUG': priority = 2

    if mtype is None and priority>1: 
        print(msg)  
    elif mtype == 'SUCCESS' and priority>1: 
        print(colored('[SUCCESS]: ', 'green') + msg)
    elif mtype == 'WARNING' and priority>0: 
        print(colored('[WARNING]: ', 'yellow') + msg)
    elif mtype == 'ERROR' and priority>0: 
        print(colored('[ERROR]: ', 'red') + msg)

### ---------------------------------------
### ------------UNIT CONVERTION------------
### ---------------------------------------

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

### ---------------------------------------
### ----------------PATHS------------------
### ---------------------------------------
paths = dict()
paths['rootDirectory'] = abspath(abspath(join(dirname(__file__), pardir)))
paths['dataDirectory'] = join(paths['rootDirectory'], 'data')
paths['interimDirectory'] = join(paths['dataDirectory'], 'interim')
paths['modelDirectory'] = join(paths['rootDirectory'], 'models')
from src.secrets import TOOLS_PATH, INVENTORY_PATH

try:
    paths['toolsDirectory'] = TOOLS_PATH
    paths['inventoryDirectory'] = INVENTORY_PATH
except:
    std_out('Cannot use inventory without path in secrets', 'WARNING')
    pass

### ---------------------------------------
### --------------BLUEPRINTS---------------
### ---------------------------------------
try:
    blueprints_path = join(paths['interimDirectory'], 'blueprints.yaml')
    with open(blueprints_path, 'r') as blueprints_yaml:
        std_out(f'Loading blueprints file from: {blueprints_path}')
        BLUEPRINTS = yaml.load(blueprints_yaml, Loader=yaml.SafeLoader)
        std_out('Loaded blueprints file')
except:
    std_out('Error loading blueprints file', 'ERROR')
    raise SystemError('Problem loading blueprints file')

def dict_fmerge(base_dct, merge_dct, add_keys=True):
    """
    From: https://gist.github.com/CMeza99/5eae3af0776bef32f945f34428669437
    Recursive dict merge.
    Args:
        base_dct (dict) onto which the merge is executed
        merge_dct (dict): base_dct merged into base_dct
        add_keys (bool): whether to add new keys
    Returns:
        dict: updated dict
    """
    rtn_dct = base_dct.copy()
    if add_keys is False:
        merge_dct = {key: merge_dct[key] for key in set(rtn_dct).intersection(set(merge_dct))}

    rtn_dct.update({
        key: dict_fmerge(rtn_dct[key], merge_dct[key], add_keys=add_keys)
        if isinstance(rtn_dct.get(key), dict) and isinstance(merge_dct[key], dict)
        else merge_dct[key]
        for key in merge_dct.keys()
    })
    return rtn_dct

for blueprint in BLUEPRINTS.keys():
    if 'expands' in BLUEPRINTS[blueprint]: 
        BLUEPRINTS[blueprint] = dict_fmerge(BLUEPRINTS[blueprint], BLUEPRINTS[BLUEPRINTS[blueprint]['expands']])
        BLUEPRINTS[blueprint].pop('expands')
std_out(f'Merged blueprints', 'SUCCESS')

### ---------------------------------------
### -------------SENSOR NAMES--------------
### ---------------------------------------
def get_firmware_names(sensorsh, json_path, file_name, reload_names = config.reload_firmware_names):
    # Directory
    names_dict = join(json_path, file_name + '.json')
    
    if reload_names:
        try:
            # Read only 20000 chars
            data = urlopen(sensorsh).read(20000).decode('utf-8')
            # split it into lines
            data = data.split('\n')
            sensor_names = dict()
            line_sensors = len(data)
            for line in data:
                if 'class AllSensors' in line:
                    line_sensors = data.index(line)
                    
                if data.index(line) > line_sensors:

                    if 'OneSensor' in line and '{' in line and '}' in line and '/*' not in line:
                        # Split commas
                        line_tokenized =  line.strip('').split(',')

                        # Elimminate unnecessary elements
                        line_tokenized_sublist = list()
                        for item in line_tokenized:
                                item = sub('\t', '', item)
                                item = sub('OneSensor', '', item)
                                item = sub('{', '', item)
                                item = sub('}', '', item)
                                #item = sub(' ', '', item)
                                item = sub('"', '', item)
                                
                                if item != '' and item != ' ':
                                    while item[0] == ' ' and len(item)>0: item = item[1:]
                                line_tokenized_sublist.append(item)
                        line_tokenized_sublist = line_tokenized_sublist[:-1]

                        if len(line_tokenized_sublist) > 2:
                                shortTitle = sub(' ', '', line_tokenized_sublist[3])
                                if len(line_tokenized_sublist)>9:
                                    sensor_names[shortTitle] = dict()
                                    sensor_names[shortTitle]['SensorLocation'] = sub(' ', '', line_tokenized_sublist[0])
                                    sensor_names[shortTitle]['id'] = sub(' ','', line_tokenized_sublist[5])
                                    sensor_names[shortTitle]['title'] = line_tokenized_sublist[4]
                                    sensor_names[shortTitle]['unit'] = line_tokenized_sublist[-1]
            # Save everything to the most recent one
            with open(names_dict, 'w') as fp:
                json.dump(sensor_names, fp)
            std_out('Saved updated sensor names and dumped into ' + names_dict, 'SUCCESS')

        except:
            # Load sensors
            print_exc()
            with open(names_dict) as handle:
                sensor_names = json.loads(handle.read())
            std_out('Error. Retrieving local version for sensors names', 'WARNING')

    else:
        std_out('Retrieving local version for sensors names')
        with open(names_dict) as handle:
            sensor_names = json.loads(handle.read())
        if sensor_names is not None: std_out('Local version of sensor names loaded', 'SUCCESS')

    return sensor_names

sensor_names_21 = get_firmware_names(config.sensor_names_url_21, join(paths['interimDirectory']), 'sensornames_21')
sensor_names_20 = get_firmware_names(config.sensor_names_url_20, join(paths['interimDirectory']), 'sensornames_20')
CURRENT_NAMES = {**sensor_names_21, **sensor_names_20}

# Update blueprints
for blueprint in BLUEPRINTS:
    # Skip non sc sensors
    if 'sc' not in blueprint[0:3]: continue
    if 'sensors' in BLUEPRINTS[blueprint]: 
        for sensor in BLUEPRINTS[blueprint]['sensors'].keys():
            BLUEPRINTS[blueprint]['sensors'][sensor]['id'] = CURRENT_NAMES[sensor]['id']
# Update file
try:
    blueprints_path = join(paths['interimDirectory'], 'blueprints.yaml')
    with open(blueprints_path, 'w') as blueprints_yaml:
        std_out(f'Updating blueprints file from: {blueprints_path}')
        yaml.dump(BLUEPRINTS, blueprints_yaml)
        std_out('Updated blueprints file', 'SUCCESS')
except:
    std_out('Error loading blueprints file', 'ERROR')
    raise SystemError('Problem saving blueprints file')

### ---------------------------------------
### ------------CALIBRATIONS---------------
### ---------------------------------------
'''
The calibrations are meant for alphasense's 4 electrode sensors.
This file follows the next structure:
{
    "Target 1": "Pollutant 1", 
    "Target 2": "Pollutant 2", 
    "Serial No": "XXXX", 
    "Sensitivity 1": "Pollutant 1 sensitivity in nA/ppm", 
    "Sensitivity 2": "Pollutant 2 sensitivity in nA/ppm", 
    "Zero Current": "in nA", 
    "Aux Zero Current": "in nA"}
}
'''
try:
    caldata_path = join(paths['interimDirectory'], 'calibrations.json')
    std_out(f'Loading calibration data from: {caldata_path}')
    CALIBRATION_DATA = pd.read_json(caldata_path, orient='columns', lines = True)
    CALIBRATION_DATA.index = CALIBRATION_DATA['serial_no']
    std_out('Loaded calibration data file', 'SUCCESS')
except:
    std_out('Error loading calibration file', 'WARNING')
    print_exc()
    pass
### ---------------------------------------
### -----------CSV LOAD/EXPORT-------------
### ---------------------------------------
def read_csv_file(file_path, location, frequency, clean_na = None, index_name = '', skiprows = None, sep = ',', encoding = 'utf-8'):
    '''
        Reads a csv file and adds cleaning, localisation and resampling
    '''

    # Read pandas dataframe
    df = pd.read_csv(file_path, verbose = False, skiprows = skiprows, sep = ',', encoding = 'utf-8')

    for column in df.columns:
        if index_name in column: 
            df = df.set_index(column)
            break

    # Set index
    df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    # Sort index
    df.sort_index(inplace=True)
    # Drop unnecessary columns
    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
    # Check for weird things in the data
    df = df.apply(pd.to_numeric,errors='coerce')   
    # # Resample
    df = df.resample(frequency, limit = 1).mean()

    # Remove na
    if clean_na is not None:
        if clean_na == 'fill':
            df = df.fillna(method='ffill')
            std_out('Filling NaN')
        elif clean_na == 'drop':
            df.dropna(axis = 0, how='all', inplace=True)
            std_out('Dropping NaN')

    return df

def export_csv_file(path, file_name, df, forced_overwrite = False):
    '''
        Exports pandas dataframe to a csv file
    '''

    # If path does not exist, create it
    if not exists(path):
        makedirs(path)

    # If file does not exist 
    if not exists(path + '/' + str(file_name) + '.csv') or forced_overwrite:
        df.to_csv(path + '/' + str(file_name) + '.csv', sep=",")
        std_out('File saved to: \n' + path + '/' + str(file_name) +  '.csv', 'SUCCESS')
    else:
        std_out("File Already exists - delete it first, I was not asked to overwrite anything!", 'WARNING')
        return False
    
    return True

### ---------------------------------------
### ------------DATE FUNCTIONS-------------
### ---------------------------------------
def get_localised_date(date, location):

    if date is not None:
        result_date = pd.to_datetime(date, utc = True)
        if result_date.tzinfo is not None: 
            result_date = result_date.tz_convert(location)
        else:
            result_date = result_date.tz_localize(location)
            print ('None')

    else: 
        result_date = None

    return result_date

### ---------------------------------------
### -------------LOC FUNCTIONS-------------
### --------------------------------------- 
def calculate_distance(location_A, location_B):
    """
    Returns distance from two locations in (lat, long, altitude)
    Parameters
    ----------
        location_A: tuple()
            (lat, long, altitude [optional])
            Latitude, longitude and altitude (optional) of first location
        location_B: tuple()
            (lat, long, altitude [optional])
            Latitude, longitude and altitude (optional) of second location
    Returns
    -------
        Distance between two points in meters
    """    
    
    return distance.distance(location_A, location_B).m