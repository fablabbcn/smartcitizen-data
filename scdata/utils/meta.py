from .dictmerge import dict_fmerge
from os import pardir, environ, name, makedirs
from os.path import join, dirname, expanduser, exists, basename
from urllib.parse import urlparse
import os
from shutil import copyfile
from requests import get
from traceback import print_exc
import json
from re import sub

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

    # - Processed data
    _paths['processed'] = join(_paths['data'], 'processed')
    makedirs(_paths['processed'], exist_ok=True)

    # - Internal data: blueprints and calibrations
    _paths['interim'] = join(_paths['data'], 'interim')
    makedirs(_paths['interim'], exist_ok=True)

    # Check for blueprints and calibrations
    # Find the path to the interim folder
    _dir = dirname(__file__)
    _idir = join(_dir, 'interim')

    # - Models and local tests
    _paths['models'] = join(_paths['data'], 'models')
    makedirs(_paths['models'], exist_ok=True)

    # - Exports
    _paths['export'] = join(_paths['data'], 'export')
    makedirs(_paths['export'], exist_ok=True)

    # - Raw
    _paths['raw'] = join(_paths['data'], 'raw')
    makedirs(_paths['raw'], exist_ok=True)
    # Copy example csvs
    _enames = ['example.csv', 'geodata.csv']
    for _ename in _enames:
        s = join(_idir, _ename)
        d = join(_paths['raw'], _ename)
        if not exists(join(_paths['raw'], _ename)): copyfile(s, d)

    # - Reports
    _paths['reports'] = join(_paths['data'], 'reports')
    makedirs(_paths['reports'], exist_ok=True)

    # - Tasks
    _paths['tasks'] = join(_paths['data'], 'tasks')
    makedirs(_paths['tasks'], exist_ok=True)

    # - Uploads
    _paths['uploads'] = join(_paths['data'], 'uploads')
    makedirs(_paths['uploads'], exist_ok=True)

    # Check for uploads
    _example_uploads = ['example_upload_1.json', 'example_zenodo_upload.yaml']
    _udir = join(_dir, 'uploads')
    for item in _example_uploads:
        s = join(_udir, item)
        d = join(_paths['uploads'], item)
        if not exists(d): copyfile(s, d)

    # Inventory (normally not used by user)
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

def load_blueprints(urls):

    blueprints = dict()
    for url in urls:
        if url is None: continue
        _nblueprint = basename(urlparse(str(url)).path).split('.')[0]
        _blueprint = get_json_from_url(url)

        if _nblueprint not in blueprints:
            blueprints[_nblueprint] = _blueprint

    return blueprints

def get_current_blueprints():
    from scdata._config import config
    if not config.is_init: config.get_meta_data()

    return list(config.blueprints.keys())

def get_json_from_url(url):

    rjson = None
    # Gets a json from an url and returns it as a dict
    try:
        rget = get(url)

        if rget.status_code == 200 or rget.status_code == 201:
            rjson = rget.json()
        else:
            print (f'Failed request. Response {rget.status_code}')
    except:
        print ('Failed request. Probably no connection or invalid json file')
        pass

    return rjson

def load_firmware_names(sensorsh):
    '''
        Loads sensor names from Sensors.h of firmware repo
        Parameters
        ----------
            sensorsh: String
                Sensors.h url
        Returns
        ---------
            Dictionary containing sensor names with descriptions
    '''
    try:
        sensor_names = dict()
        # Read only 20000 chars
        data = get(sensorsh).text
        # split it into lines
        data = data.split('\n')
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
    except:
        print ('Problem while loading Sensors.h file')
        print_exc()
        sensor_names = None
        pass
    return sensor_names

def load_calibrations(urls):
    '''
        Loads calibrations from urls.
        The calibrations are meant for alphasense's 4 electrode sensors. The files contains:
        {
          "162031254": {
            "ae_electronic_zero_mv": "",
            "ae_sensor_zero_mv": "-16.64",
            "ae_total_zero_mv": "",
            "pcb_gain_mv_na": "0.8",
            "we_cross_sensitivity_no2_mv_ppb": "0",
            "we_cross_sensitivity_no2_na_ppb": "0",
            "we_electronic_zero_mv": "",
            "we_sensitivity_mv_ppb": "0.45463999999999993",
            "we_sensitivity_na_ppb": "0.5682999999999999",
            "we_sensor_zero_mv": "-27.200000000000003",
            "we_total_zero_mv": ""
          },
        ...
        }
        Parameters
        ----------
            path: String
                json file path
        Returns
        ---------
            Dictionary containing calibrations otherwise None
    '''

    calibrations = dict()
    for url in urls:
        try:
            calibrations = dict_fmerge(get_json_from_url(url), calibrations)
        except:
            print(f'Problem loading calibrations from {url}')
            return None

    return calibrations

def load_connectors(urls):

    connectors = dict()
    for url in urls:
        try:
            c = get_json_from_url(url)
            _nc = basename(urlparse(str(url)).path).split('.')[0]
            connectors[_nc] = c
        except:
            print(f'Problem loading connectors from {url}')
            print_exc()
            return None

    return connectors
