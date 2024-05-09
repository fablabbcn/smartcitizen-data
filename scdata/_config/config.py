import yaml
import json
from os import pardir, environ
from os.path import join, abspath, dirname, exists
import sys
from math import inf
from numpy import array
import logging
from os import pardir, environ, name, makedirs
from os.path import join, dirname, expanduser, exists, basename
from urllib.parse import urlparse
import os
from shutil import copyfile
from requests import get
# from traceback import print_exc
import json
from pydantic import TypeAdapter
from typing import List

from scdata.models import Name, Blueprint, Metric
from scdata.tools.dictmerge import dict_fmerge
from scdata.tools.gets import get_json_from_url

class Config(object):
    ### ---------------------------------------
    ### ---------------LOG-LEVEL---------------
    ### ---------------------------------------
    _log_level = logging.INFO

    # Framework option
    # For renderer plots and config files
    # Options:
    # - 'script': no plots in jupyter, updates config
    # - 'jupyterlab': for plots, updates config
    # - 'chupiflow': no plots in jupyter, does not update config
    framework = 'script'

    if 'IPython' in sys.modules: _ipython_avail = True
    else: _ipython_avail = False

    # Returns when iterables cannot be fully processed
    _strict = False

    # Timeout for http requests
    _timeout = 3
    _max_http_retries = 2

    # Max concurrent requests
    _max_concurrent_requests = 30

    ### ---------------------------------------
    ### -----------------DATA------------------
    ### ---------------------------------------

    data = {
        # Whether or not to reload metadata from git repo
        'reload_metadata': True,
        # Whether or not to load or store cached data (saves time when requesting a lot of data)
        'load_cached_api': True,
        'store_cached_api': True,
        # If reloading data from the API, how much gap between the saved data and the
        # latest reading in the API should be ignore
        'cached_data_margin': 1,
        # clean_na
        'clean_na': None,
        # Ignore additional channels from API or CSV that are not in the blueprint.json
        'strict_load': False
    }

    # Maximum amount of points to load when postprocessing data
    _max_load_amount = 500

    # Ingore Nas when loading data (for now only in CSVs)
    # Similar to na_values in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    _ignore_na_values = [' nan']

    ### ---------------------------------------
    ### --------------ALGORITHMS---------------
    ### ---------------------------------------

    ## TODO - move out from here
    # Whether or not to plot intermediate debugging visualisations in the algorithms
    _intermediate_plots = False

    # Plot out level (priority of the plot to show - 'DEBUG' or 'NORMAL')
    _plot_out_level = 'DEBUG'

    ### ---------------------------------------
    ### ----------------ZENODO-----------------
    ### ---------------------------------------

    # Urls
    zenodo_sandbox_base_url='http://sandbox.zenodo.org'
    zenodo_real_base_url='https://zenodo.org'

    ### ---------------------------------------
    ### -------------SMART CITIZEN-------------
    ### ---------------------------------------
    # # Urls
    _base_postprocessing_url = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/enhacement/flexible-handlers/'
    _default_file_type = 'json'

    calibrations_urls = [
        f'{_base_postprocessing_url}calibrations/calibrations.{_default_file_type}'
    ]

    blueprints_urls = [
        # f'{_base_postprocessing_url}blueprints/base.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/csic_station.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/muv_station.{_default_file_type}',
        # # f'{_base_postprocessing_url}blueprints/parrot_soil.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sc_20_station_iscape.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sc_21_station_iscape.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sc_21_station_module.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_15.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_20.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sc_air.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sc_water.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_21_sps30.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_21_sen5x.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_21_gps.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_21_nilu.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sck_21_co2.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/sc_21_water.{_default_file_type}'
    ]

    # connectors_urls = [
    #     f'{_base_postprocessing_url}connectors/nilu.{_default_file_type}'
    # ]

    names_urls = [
        # Revert to base postprocessing url
        # f'{_base_postprocessing_url}names/SCDevice.json'
        'https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/enhacement/flexible-handlers/names/SCDevice.json'
    ]


    ### ---------------------------------------
    ### -------------METRICS DATA--------------
    ### ---------------------------------------
    # Metrics levels
    _channel_bins = {
        'NOISE': [-inf, 52, 54, 56, 58, 60, 62, 64, 66, 68, inf],
        'PM': [-inf, 10, 20, 30, 40, 50, 75, 100, 150, 200, inf],
        'GPS_HDOP': [-inf, 0, 40, 80, 120, 160, 200, 240, 260, 300, inf]
    }


    _channel_bin_n = 11

    ### ---------------------------------------
    ### ----------------PLOTS------------------
    ### ---------------------------------------
    _plot_def_opt = {
        'min_date': None,
        'max_date': None,
        'frequency': None,
        'resample': 'mean',
        'clean_na': None,
        'show': True
    }

    _map_def_opt = {
        'location': [41.400818, 2.1825157],
        'tiles': 'cartodbpositron',
        'zoom': 2.5,
        'period': '1W',
        'radius': 10,
        'fillOpacity': 1,
        'stroke': 'false',
        'icon': 'circle',
        'minmax': False,
        'max_speed': 10,
        'minimap': True,
        'legend': True,
        'markers': True
    }

    _map_colors_palette = array(['#053061','#2166ac','#4393c3','#92c5de',
                                 '#d1e5f0','#fddbc7','#f4a582','#d6604d',
                                 '#b2182b','#67001f'])

    _plot_style = "seaborn-v0_8-whitegrid"

    _ts_plot_def_fmt = {
        'mpl': {
            'width': 12,
            'height': 8,
            'sharex': True,
            'ylabel': None,
            'xlabel': 'Date (-)',
            'yrange': None,
            'xrange': None,
            'title': None,
            'suptitle_factor': 0.92,
            'suptitle_x': 0.5,
            'suptitle_y': 0.98,
            'title_loc': "center",
            'title_fontsize': 14,
            'hspace':0.4,
            'wspace':0.4,
            'grid': True,
            'fontsize': 10,
            'alpha_highlight': 0.8,
            'alpha_other': 0.1,
            'alpha_bands': 0.2,
            'palette': None,
            'decorators': None,
            'legend': True,
            'style': _plot_style
        },
        'plotly': {
            'width': 800,
            'height': 600,
            'sharex': True,
            'ylabel': None,
            'yrange': None,
            'xlabel': 'Date (-)',
            'title': None,
            'suptitle_factor': 0.92,
            'grid': True,
            'fontsize': 13,
            'alpha_highlight': 0.8,
            'alpha_other': 0.1,
            'palette': None,
            'decorators': None,
            'legend': True
        },
        'uplot': {
            'width': 800,
            'height': 400,
            'ylabel': None,
            'size': 3,
            'xlabel': 'Date (-)',
            'title': None,
            'padding-right': 50,
            'padding-bottom': 200,
            'fontsize': 15,
            'legend_isolate': True
        }
    }

    _scatter_plot_def_fmt = {
        'mpl': {
            'height': 10,
            'width': 12,
            'ylabel': None,
            'xlabel': None,
            'yrange': None,
            'xrange': None,
            'title': None,
            'suptitle_factor': 0.92,
            'grid': True,
            'fontsize': 10,
            'title_fontsize': 14,
            'palette': None,
            'legend': True,
            'style': _plot_style,
            'kind': 'reg',
            'sharex': False,
            'sharey': False,
            'nrows': 1
        },
        'plotly': {
            'height': 600,
            # 'ylabel': None, TODO Not yet
            # 'xlabel': None, TODO Not yet
            'title': None,
            # 'suptitle_factor': 0.92, TODO Not yet
            'grid': True,
            'fontsize': 10,
            'title_fontsize': 14,
            # 'decorators': None, TODO Not yet
            'legend': True,
            'kind': 'scatter'
        }
    }

    _ts_scatter_def_fmt = {
        'mpl': {
            'width': 24,
            'height': 8,
            'ylabel': None,
            'xlabel': 'Date (-)',
            'yrange': None,
            'xrange': None,
            'title': None,
            # 'suptitle_factor': 0.92, TODO Not yet
            'grid': True,
            'fontsize': 10,
            'title_fontsize': 14,
            # 'decorators': None, TODO Not yet
            'legend': True,
            'style': _plot_style
        }
    }

    _heatmap_def_fmt = {
        'mpl': {
            'height': 10,
            'width': 20,
            'xlabel': 'Date (-)',
            'yrange': None,
            'xrange': None,
            'title': None,
            'suptitle_factor': 0.92,
            'grid': True,
            'fontsize': 10,
            'title_fontsize': 14,
            'cmap': 'RdBu_r',
            'legend': True,
            'style': _plot_style,
            'robust': True,
            'vmin': None,
            'vmax': None,
            'frequency_hours': 2,
            'session': '1D'
        },
        'plotly': {
            'height': 600,
            'width': 800,
            'xlabel': 'Date (-)',
            'yrange': None,
            'xrange': None,
            'title': None,
            'grid': True,
            # 'fontsize': 10,
            # 'title_fontsize': 14,
            # 'cmap': 'RdBu_r',
            'legend': True,
            # 'robust': True,
            # 'vmin': None,
            # 'vmax': None,
            'frequency_hours': 2,
            'session': '1D'
        }
    }

    _boxplot_def_fmt = {
        'mpl': {
            'height': 10,
            'width': 20,
            'ylabel': None,
            'yrange': None,
            'title': None,
            'suptitle_factor': 0.92,
            'grid': True,
            'fontsize': 10,
            'title_fontsize': 14,
            'cmap': 'RdBu_r',
            'style': _plot_style,
            'frequency_hours': 2,
            'session': '1D',
            'palette': None,
            'periods': None
        }
    }

    _dendrogram_def_fmt = {
        'mpl': {
            'height': 10,
            'width': 25,
            'ylabel': 'Name',
            'xlabel': 'Distance',
            'title': 'Hierarchical Clustering dendrogram',
            'fontsize': 8.,
            'orientation': 'left',
            'title_fontsize': 14,
            'suptitle_factor': 0.92,
            'style': _plot_style,
            'palette': None
        }
    }

    _missingno_def_fmt = {
        'height': 6,
        'width': 6,
        'fontsize': 8.,
        'title_fontsize': 14
    }


    ### ---------------------------------------
    ### ----------------MODELS-----------------
    ### ---------------------------------------

    _model_def_opt = {
        'test_size': 0.2,
        'shuffle': False,
        'clean_na': 'drop',
        'common_avg': False
    }

    _model_hyperparameters = {
        'rf': {
            'n_estimators': 100,
            'min_samples_leaf': 2,
            'max_features': None,
            'oob_score': True
        }
    }

    ### ---------------------------------------
    ### --------------DISPERSION---------------
    ### ---------------------------------------

    _dispersion = {
        # Use average dispersion or instantaneous
        'instantatenous_dispersion': False,
        # Percentage of points to be considered NG sensor
        'limit_errors': 3,
        # Multiplier for std_dev (sigma) - Normal distribution (99.73%)
        'limit_confidence_sigma': 3,
        # t-student confidence level (%)
        't_confidence_level': 99,
        # In case there is a device with lower amount of channels, ignore the missing channels and keep going
        'ignore_missing_channels': True,
        # Do not perform dispersion analysis on these channels
        'ignore_channels': ['BATT'],
        # Normal or t-student distribution threshold
        'nt_threshold': 30
    }

    ### ---------------------------------------
    ### ----------------CSV--------------------
    ### ---------------------------------------

    _csv_defaults = {
        'index_name': 'TIME',
        'sep': ',',
        'skiprows': None
    }

    ### ---------------------------------------
    ### ---------------------------------------
    ### ---------------------------------------

    def __init__(self):
        self._env_file = None
        self.paths = self.get_paths()
        self.load()
        self.get_meta_data()


    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError

    def __getitem__(self, key):
        try:
            val = self.__getattribute__(key)
        except KeyError:
            raise KeyError
        else:
            return val

    def __iter__(self):
        return (i for i in dir(self))

    def load_env(self):
        with open(self._env_file) as f:
            for line in f:
                # Ignore empty lines or lines that start with #
                if line.startswith('#') or not line.strip():
                    continue
                # Load to local environ
                key, value = line.strip().split('=', 1)
                environ[key] = value

    def load_calibrations(self, urls):
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
                urls: [String]
                    json file urls
            Returns
            ---------
                Dictionary containing calibrations otherwise None
        '''

        calibrations = dict()
        for url in urls:
            try:
                rjson, _ = get_json_from_url(url)
                calibrations = dict_fmerge(rjson, calibrations)
            except:
                print(f'Problem loading calibrations from {url}')
                return None

        return calibrations

    # def load_connectors(self, urls):
    #     connectors = dict()
    #     for url in urls:
    #         try:
    #             c = get_json_from_url(url)
    #             _nc = basename(urlparse(str(url)).path).split('.')[0]
    #             connectors[_nc] = c
    #         except:
    #             print(f'Problem loading connectors from {url}')
    #             print_exc()
    #             return None

    #     return connectors

    def load_blueprints(self, urls):
        blueprints = dict()
        for url in urls:
            if url is None: continue
            _nblueprint = basename(urlparse(str(url)).path).split('.')[0]
            rjson, _ = get_json_from_url(url)

            if rjson is None:
                continue
            if _nblueprint not in blueprints:
                blueprints[_nblueprint] = TypeAdapter(Blueprint).validate_python(rjson).model_dump()

        return blueprints

    def load_names(self, urls):
        isn = True
        names = dict()

        for url in urls:
            result = list()
            _nc = basename(urlparse(str(url)).path).split('.')[0]

            while isn:
                try:
                    rjson, rheaders = get_json_from_url(url)
                    result += TypeAdapter(List[Name]).validate_python(rjson)
                except:
                    isn = False
                    pass
                else:
                    if 'next' in rheaders:
                        if rheaders['next'] == url: isn = False
                        elif rheaders['next'] != url: url = rheaders['next']
                    else:
                        isn = False
            names[_nc] = result

        return names

    def get_paths(self):

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
        _idir = join(_dir, '../tools/interim')

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
        _udir = join(_dir, '../tools/uploads')
        for item in _example_uploads:
            s = join(_udir, item)
            d = join(_paths['uploads'], item)
            if not exists(d): copyfile(s, d)

        # Inventory (normally not used by user)
        _paths['inventory'] = ''

        return _paths

    def get_meta_data(self):
        """ Get meta data from blueprints and _calibrations """

        # Load blueprints, calibrations and names
        bppath = join(self.paths['interim'], 'blueprints.json')
        if self.data['reload_metadata'] or not exists(bppath):
            blueprints = self.load_blueprints(self.blueprints_urls)
            bpreload = True
        else:
            with open(bppath, 'r') as file: blueprints = json.load(file)
            bpreload = False

        calpath = join(self.paths['interim'], 'calibrations.json')
        if self.data['reload_metadata'] or not exists(calpath):
            calibrations = self.load_calibrations(self.calibrations_urls)
            calreload = True
        else:
            with open(calpath, 'r') as file: calibrations = json.load(file)
            calreload = False

        namespath = join(self.paths['interim'], 'names.json')
        if self.data['reload_metadata'] or not exists(namespath):
            names = self.load_names(self.names_urls)
            namesreload = True
        else:
            names = dict()
            with open(namespath, 'r') as file:
                names_load = json.load(file)
            for item in names_load:
                names[item] = TypeAdapter(List[Name]).validate_python(names_load[item])
            namesreload = False

        # Dump blueprints, calibrations and names
        if blueprints is not None:
            self.blueprints = blueprints
            if bpreload:
                with open(bppath, 'w') as file:
                    json.dump(blueprints, file)

        if calibrations is not None:
            self.calibrations = calibrations
            if calreload:
                with open(calpath, 'w') as file:
                    json.dump(calibrations, file)

        if names is not None:
            self.names = names
            if namesreload:
                for item in self.names:
                    names_dump = {item: [name.model_dump() for name in self.names[item]]}
                with open(namespath, 'w') as file:
                    json.dump(names_dump, file)

        # Find environment file in root or in scdata/ for clones
        if exists(join(self.paths['data'],'.env')):
            self._env_file = join(self.paths['data'],'.env')
            print(f'Found Environment file at: {self._env_file}')
            self.load_env()
        else:
            print(f'No environment file found. If you had an environment file (.env) before, make sure its now here')
            print(join(self.paths['data'],'.env'))

    def load(self):
        """ Override config if config file exists. """
        _sccpath = join(self.paths['config'], 'config.yaml')

        # Thankfully inspired in config.py by mps-youtube
        if exists(_sccpath):
            with open(_sccpath, "r") as cf:
                saved_config = yaml.load(cf, Loader = yaml.SafeLoader)

            for k, v in saved_config.items():

                try:
                    self.__setattr__(k, v)

                except KeyError:  # Ignore unrecognised data in config
                    print ("Unrecognised config item: %s", k)

        if self.framework != 'chupiflow':
            self.save()

    def save(self):
        """ Save current config to file. """
        c = dict()
        for setting in self:
            if not setting.startswith('_') and not callable(self.__getitem__(setting)) and setting not in ['blueprints', 'names', 'calibrations']:
                c[setting] = self[setting]

        _sccpath = join(self.paths['config'], 'config.yaml')
        with open(_sccpath, "w") as cf:
            yaml.dump(c, cf)

