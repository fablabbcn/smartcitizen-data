import yaml
import json

from scdata.utils.dictmerge import dict_fmerge
from scdata.utils.meta import (get_paths, load_blueprints,
                                load_calibrations, load_connectors,
                                load_env, load_firmware_names)

from os import pardir, environ
from os.path import join, abspath, dirname, exists
import sys

from math import inf
from numpy import arange, array

class Config(object):

    # Output level. 'QUIET': nothing, 'NORMAL': warn, err,
    # 'DEBUG': info, warn, err
    _out_level = 'NORMAL'

    # Timestamp for log output
    _timestamp = True

    # Framework option
    # For renderer plots and config files
    # Options:
    # - 'script': no plots in jupyter, updates config
    # - 'jupyterlab': for plots, updates config
    # - 'chupiflow': no plots in jupyter, does not update config
    framework = 'script'

    if 'IPython' in sys.modules: _ipython_avail = True
    else: _ipython_avail = False

    # Default location for timezones
    _location = 'Europe/Madrid'
    _epsg = 4326

    # Returns when iterables cannot be fully processed
    _strict = False

    ### ---------------------------------------
    ### ----------------CRONTAB----------------
    ### ---------------------------------------
    # Tabfile for cronjobs
    _tabfile = 'tabfile'

    # Scheduler
    _scheduler_interval_days = 1
    _device_scheduler = 'dschedule'
    _scheduler_log = 'scheduler.log'
    # Tasks
    _postprocessing_interval_hours = 1
    _device_processor = 'dprocess'

    ### ---------------------------------------
    ### -----------------DATA------------------
    ### ---------------------------------------

    ## Place here options for data load and handling
    _combined_devices_name = 'COMBINED_DEVICES'

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
        'clean_na': None
    }

    # If using multiple training datasets, how to call the joint df
    _name_multiple_training_data = 'CDEV'

    # Maximum amount of points to load when postprocessing data
    _max_load_amount = 500

    # Ignore duplicate sensor ids
    _sc_ignore_keys = ['DALLAS_TEMP', 'GB_TEMP', 'GB_HUM']

    ### ---------------------------------------
    ### --------------ALGORITHMS---------------
    ### ---------------------------------------

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
    sensor_names_url_21='https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-21/master/lib/Sensors/Sensors.h'
    # sensor_names_url_20='https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/master/lib/Sensors/Sensors.h'
    _base_postprocessing_url = 'https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/master/'
    _default_file_type = 'json'

    calibrations_urls = [
        f'{_base_postprocessing_url}calibrations/calibrations.{_default_file_type}'
    ]

    blueprints_urls = [
        f'{_base_postprocessing_url}blueprints/base.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/csic_station.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/muv_station.{_default_file_type}',
        # f'{_base_postprocessing_url}blueprints/parrot_soil.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sc_20_station_iscape.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sc_21_station_iscape.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sc_21_station_module.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck_15.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck_20.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck_21.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck_21_gps.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck_21_nilu.{_default_file_type}',
        f'{_base_postprocessing_url}blueprints/sck_21_co2.{_default_file_type}'
    ]

    connectors_urls = [
        f'{_base_postprocessing_url}connectors/nilu.{_default_file_type}'
    ]

    # Convertion table from API SC to Pandas
    # https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
    # https://developer.smartcitizen.me/#get-historical-readings
    _freq_conv_lut = (
        ['y','A'],
        ['M','M'],
        ['w','W'],
        ['d','D'],
        ['h','H'],
        ['m','Min'],
        ['s','S'],
        ['ms','ms']
    )

    # AlphaDelta PCB factor (converstion from mV to nA)
    _alphadelta_pcb = 6.36
    # Deltas for baseline deltas algorithm
    _baseline_deltas = arange(30, 45, 5)
    # Lambdas for baseline ALS algorithm
    _baseline_als_lambdas = [1e5]

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

    # Molecular weights of certain pollutants for unit convertion
    _molecular_weights = {
        'CO':   28,
        'NO':   30,
        'NO2':  46,
        'O3':   48,
        'C6H6': 78,
        'SO2':  64,
        'H2S':  34
    }

    # Background concentrations
    _background_conc = {
        'CO':   0,
        'NO2':  8,
        'O3':   40
    }

    # Alphasense data
    _as_sensor_codes =  {
        '132':  'ASA4_CO',
        '133':  'ASA4_H2S',
        '130':  'ASA4_NO',
        '212':  'ASA4_NO2',
        '214':  'ASA4_OX',
        '134':  'ASA4_SO2',
        '162':  'ASB4_CO',
        '133':  'ASB4_H2S',#
        '130':  'ASB4_NO', #
        '202':  'ASB4_NO2',
        '204':  'ASB4_OX',
        '164':  'ASB4_SO2'
    }

    # From Tables 2 and 3 of AAN 803-04
    _as_t_comp = [-30, -20, -10, 0, 10, 20, 30, 40, 50]

    _as_sensor_algs = {
        'ASA4_CO':
                    {
                        1: ['n_t',      [1.0, 1.0, 1.0, 1.0, -0.2, -0.9, -1.5, -1.5, -1.5]],
                        4: ['kpp_t',    [13, 12, 16, 11, 4, 0, -15, -18, -36]],
                    },
        'ASA4_H2S':
                    {
                        2: ['k_t',      [-1.5, -1.5, -1.5, -0.5, 0.5, 1.0, 0.8, 0.5, 0.3]],
                        1: ['n_t',      [3.0, 3.0, 3.0, 1.0, -1.0, -2.0, -1.5, -1.0, -0.5]]
                    },
        'ASA4_NO':
                    {
                        3: ['kp_t',     [0.7, 0.7, 0.7, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6]],
                        4: ['kpp_t',    [-25, -25, -25, -25, -16, 0, 56, 200, 615]]
                    },
        'ASA4_NO2':
                    {
                        1: ['n_t',      [0.8, 0.8, 1.0, 1.2, 1.6, 1.8, 1.9, 2.5, 3.6]],
                        3: ['kp_t',     [0.2, 0.2, 0.2, 0.2, 0.7, 1.0, 1.3, 2.1, 3.5]]
                    },
        'ASA4_OX':
                    {
                        3: ['kp_t',     [0.1, 0.1, 0.2, 0.3, 0.7, 1.0, 1.7, 3.0, 4.0]],
                        1: ['n_t',      [1.0, 1.2, 1.2, 1.6, 1.7, 2.0, 2.1, 3.4, 4.6]]
                    },
        'ASA4_SO2':
                    {
                        4: ['kpp_t',    [0, 0, 0, 0, 0, 0, 5, 25, 45]],
                        1: ['n_t',      [1.3, 1.3, 1.3, 1.2, 0.9, 0.4, 0.4, 0.4, 0.4]]
                    },
        'ASB4_CO':
                    {
                        1: ['n_t',      [0.7, 0.7, 0.7, 0.7, 1.0, 3.0, 3.5, 4.0, 4.5]],
                        2: ['k_t',      [0.2, 0.2, 0.2, 0.2, 0.3, 1.0, 1.2, 1.3, 1.5]]
                    },
        'ASB4_H2S':
                    {
                        1: ['n_t',      [-0.6, -0.6, 0.1, 0.8, -0.7, -2.5, -2.5, -2.2, -1.8]],
                        2: ['k_t',      [0.2, 0.2, 0.0, -0.3, 0.3, 1.0, 1.0, 0.9, 0.7]]
                    },
        'ASB4_NO':
                    {
                        2: ['k_t',      [1.8, 1.8, 1.4, 1.1, 1.1, 1.0, 0.9, 0.9, 0.8]],
                        3: ['kp_t',     [0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
                    },
        'ASB4_NO2':
                    {
                        1: ['n_t',      [1.3, 1.3, 1.3, 1.3, 1.0, 0.6, 0.4, 0.2, -1.5]],
                        3: ['kp_t',     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, -0.1, -4.0]]
                    },
        'ASB4_OX':
                    {
                        1: ['n_t',      [0.9, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.7]],
                        3: ['kp_t',     [0.5, 0.5, 0.5, 0.6, 0.6, 1.0, 2.8, 5.0, 5.3]]
                    },
        'ASB4_SO2':
                    {
                        4: ['kpp_t',    [-4, -4, -4, -4, -4, 0, 20, 140, 450]],
                        1: ['n_t',      [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.9, 3.0, 5.8]]
                    }
        }

    # This look-up table is comprised of channels you want always want to have with the same units and that might come from different sources
    # i.e. pollutant data in various units (ppm or ug/m3) from different analysers
    # The table should be used as follows:
    # 'key': 'units',
    # - 'key' is the channel that will lately be used in the analysis. It supports regex
    # - target_unit is the unit you want this channel to be and that will be converted in case of it being found in the channels list of your source

    _channel_lut = {
        "TEMP": "degC",
        "HUM": "%rh",
        "PRESS": "kPa",
        "PM_(\d|[A,B]_\d)": "ug/m3",
        "CO(\D|$)": "ppb",
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

    _unit_convertion_lut = (
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


    _plot_style = "seaborn-whitegrid"

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
            'grid': True,
            'fontsize': 10,
            'title_fontsize': 14,
            'alpha_highlight': 0.8,
            'alpha_other': 0.1,
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
            'suptitle_factor': 0.92
        }
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
    ### ----------------DISCARD----------------
    ### ---------------------------------------

    _discvars = [
        'readings',
        'api_device',
        'options',
        'loaded',
        'hw_id',
        'blueprint_url',
        'hardware_url',
        'processed',
        'forwarding_params',
        'meta',
        'processed',
        'postprocessing_info',
        'hw_updated_at',
        'description',
        'latest_postprocessing',
        'blueprint_loaded_from_url',
        'hardware_loaded_from_url'
    ]

    def __init__(self):
        self._env_file = False
        self.paths = get_paths()
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

    def get_meta_data(self):
        """ Get meta data from blueprints and _calibrations """

        # (re)load calibrations
        bppath = join(self.paths['interim'], 'blueprints.json')
        if self.data['reload_metadata'] or not exists(bppath):
            blueprints = load_blueprints(self.blueprints_urls)
            bpreload = True
        else:
            with open(bppath, 'r') as file: blueprints = json.load(file)
            bpreload = False

        calpath = join(self.paths['interim'], 'calibrations.json')
        if self.data['reload_metadata'] or not exists(calpath):
            calibrations = load_calibrations(self.calibrations_urls)
            calreload = True
        else:
            with open(calpath, 'r') as file: calibrations = json.load(file)
            calreload = False

        conpath = join(self.paths['interim'], 'connectors.json')
        if self.data['reload_metadata'] or not exists(conpath):
            connectors = load_connectors(self.connectors_urls)
            conreload = True
        else:
            with open(conpath, 'r') as file: connectors = json.load(file)
            conreload = False

        namespath = join(self.paths['interim'], 'names.json')
        if self.data['reload_metadata'] or not exists(namespath):
            sc_sensor_names = load_firmware_names(self.sensor_names_url_21)
            namesreload = True
        else:
            with open(namespath, 'r') as file: sc_sensor_names = json.load(file)
            namesreload = False

        if blueprints is not None:
            self.blueprints = blueprints
            if bpreload:
                with open(bppath, 'w') as file: json.dump(blueprints, file)
        if calibrations is not None:
            self.calibrations = calibrations
            if calreload:
                with open(calpath, 'w') as file: json.dump(calibrations, file)
        if connectors is not None:
            self.connectors = connectors
            if conreload:
                with open(conpath, 'w') as file: json.dump(connectors, file)
        if sc_sensor_names is not None:
            self.sc_sensor_names = sc_sensor_names
            if namesreload:
                with open(namespath, 'w') as file: json.dump(sc_sensor_names, file)

        # Find environment file in root or in scdata/ for clones
        if exists(join(self.paths['data'],'.env')):
            env_file = join(self.paths['data'],'.env')
        else:
            print (f'No environment file found. \
                    If you had an environment file (.env) before, \
                    make sure its now here:')
            print(join(self.paths['data'],'.env'))
            env_file = None

        # Load .env for tokens and stuff if found
        if env_file is not None and not self._env_file:
            print(f'Found Environment file at: {env_file}')
            if load_env(env_file): self._env_file = True

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
            if not setting.startswith('_') and not callable(self.__getitem__(setting)):
                c[setting] = self[setting]

        _sccpath = join(self.paths['config'], 'config.yaml')
        with open(_sccpath, "w") as cf:
            yaml.dump(c, cf)

    def set_testing(self, env_file = None):
        '''
        Convenience method for setting variables as development
        in jupyterlab
        Parameters
        ----------
            None
        Returns
        ----------
            None
        '''

        print ('Setting test mode')
        self._out_level = 'DEBUG'
        self.framework = 'jupyterlab'
        self._intermediate_plots = True
        self._plot_out_level = 'DEBUG'

        # Load Environment
        if env_file is not None and not self._env_file:
            if load_env(env_file): self._env_file = True
