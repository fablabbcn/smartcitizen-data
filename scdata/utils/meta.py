from os.path import join
import yaml
from .dictmerge import dict_fmerge
from pandas import read_json
from os import pardir, environ
from os.path import join, abspath, dirname

def get_paths():

    # Try to initialise paths based on environment
    try:
        _paths = dict()
        if 'DATA_PATH' in environ.keys():
            _paths['data'] = environ['DATA_PATH']
            _paths['processed'] = join(_paths['data'], 'processed')
            _paths['interim'] = join(_paths['data'], 'interim')
            _paths['models'] = join(_paths['data'], 'models')
            # print ('DATA_PATH found in environ')
        else: 
            print ('DATA_PATH not set in environ')
        
        if 'INVENTORY_PATH' in environ.keys():
            _paths['inventory'] = environ['INVENTORY_PATH']
            # print ('INVENTORY_PATH found in environ')
        else: 
            print ('INVENTORY_PATH not set in environ')        

        if 'TOOLS_PATH' in environ.keys(): 
            _paths['tools'] = environ['TOOLS_PATH']
            # print ('TOOLS_PATH found in environ')
        else: 
            print ('TOOLS_PATH not set in environ')            
    except:
        return None
   
    else:
        return _paths

def load_env(env_file):
    try:
        with open(env_file) as f:
            for line in f:
                # Ignore empty lines or lines that start with #
                if line.startswith('#') or not line.strip(): continue
                # Load to local environ
                key, value = line.strip().split('=', 1)
                environ[key] = value  
        return True
    except FileNotFoundError:
        print('.env file not found')
        return False

def load_blueprints(paths):
    
    try:
        blueprints_path = join(paths['interim'], 'blueprints.yaml')
        with open(blueprints_path, 'r') as blueprints_yaml:
            blueprints = yaml.load(blueprints_yaml, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        print('Problem loading blueprints file')
        return None
    else:
        for blueprint in blueprints.keys():
            if 'expands' in blueprints[blueprint]: 
                blueprints[blueprint] = dict_fmerge(blueprints[blueprint], blueprints[blueprints[blueprint]['expands']])
                blueprints[blueprint].pop('expands')
            
        return blueprints

# TODO
def add_blueprint(**kwargs):
    print ('not yet')

def get_current_blueprints():
    from scdata._config import config
    if not config.is_init: config.get_meta_data()

    return list(config.blueprints.keys())

def load_calibrations(paths):
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
        caldata_path = join(paths['interim'], 'calibrations.json')
        caldf = read_json(caldata_path, orient='columns', lines = True)
        caldf.index = caldf['serial_no']
    except FileNotFoundError:
        print('Problem loading calibrations file')
        return None
    else:   
        return caldf