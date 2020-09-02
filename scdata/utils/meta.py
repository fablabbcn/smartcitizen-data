from os.path import join
import yaml
from .dictmerge import dict_fmerge
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
    print ('Not yet')

def get_current_blueprints():
    from scdata._config import config
    if not config.is_init: config.get_meta_data()

    return list(config.blueprints.keys())

def load_calibrations(paths):
    '''
        Loads calibrations from yaml file. 
        The calibrations are meant for alphasense's 4 electrode sensors. The yaml file contains:
        'SENSOR_ID':
            sensor_type: ''
            we_electronic_zero_mv: ''
            we_sensor_zero_mv: ''
            we_total_zero_mv: ''
            ae_electronic_zero_mv: ''
            ae_sensor_zero_mv: ''
            ae_total_zero_mv: ''
            we_sensitivity_na_ppb: ''
            we_cross_sensitivity_no2_na_ppb: ''
            pcb_gain: ''
            we_sensitivity_mv_ppb: ''
            we_cross_sensitivity_no2_mv_ppb: ''
        Parameters
        ----------
            path: String
                yaml file path
        Returns
        ---------
            Dictionary containing calibrations otherwise None
    '''
    try:
        calspath = join(paths['interim'], 'calibrations.yaml')
        
        with open(calspath, 'r') as j:
            cals = load.load(j)
    except FileNotFoundError:
        print('Problem loading calibrations file')
        return None
    else:   
        return cals