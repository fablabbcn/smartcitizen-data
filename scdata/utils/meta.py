import yaml
from .dictmerge import dict_fmerge
from os import pardir, environ, name, makedirs
from os.path import join, abspath, dirname, expanduser, exists
import os

def get_paths():

    # Check if windows
    _mswin = name == "nt"
    # Get user_home
    _user_home = expanduser("~")

    # Get .config dir
    if _mswin:
        _cdir = environ["APPDATA"]
    elif 'XDG_CONFIG_HOME' in environ:
        _cdir = environ['XDG_CONFIG_HOME']
    else:
        _cdir = join(expanduser("~"), '.config')

    # Get .cache dir - maybe change it if found in config.json
    if _mswin:
        _ddir = environ["APPDATA"]
    elif 'XDG_CACHE_HOME' in environ:
        _ddir = environ['XDG_CACHE_HOME']
    else:
        _ddir = join(expanduser("~"), '.cache')

    # Set config and cache (data) dirs
    _sccdir = join(_cdir, 'scdata')
    _scddir = join(_ddir, 'scdata')

    makedirs(_sccdir, exist_ok=True)
    makedirs(_scddir, exist_ok=True)

    _paths = dict()

    _paths['config'] = _sccdir
    _paths['data'] = _scddir

    # Auxiliary folders
    _paths['processed'] = join(_paths['data'], 'processed')
    makedirs(_paths['processed'], exist_ok=True)
    _paths['interim'] = join(_paths['data'], 'interim')
    makedirs(_paths['interim'], exist_ok=True)
    _paths['models'] = join(_paths['data'], 'models')
    makedirs(_paths['models'], exist_ok=True)
    _paths['export'] = join(_paths['data'], 'export')
    makedirs(_paths['export'], exist_ok=True)
    _paths['raw'] = join(_paths['data'], 'raw')
    makedirs(_paths['raw'], exist_ok=True)
    _paths['reports'] = join(_paths['data'], 'reports')
    makedirs(_paths['reports'], exist_ok=True)
    _paths['uploads'] = join(_paths['data'], 'uploads')
    makedirs(_paths['uploads'], exist_ok=True)

    _paths['inventory'] = ''

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
    
    except FileNotFoundError:
        print('.env file not found')
        return False
    else:
        return True

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
            cals = yaml.load(j, Loader = yaml.SafeLoader)
    except FileNotFoundError:
        print('Problem loading calibrations file')
        return None
    else:   
        return cals