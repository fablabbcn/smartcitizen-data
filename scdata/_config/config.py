from os.path import join
import yaml

from scdata.utils.dictmerge import dict_fmerge
from scdata.utils.meta import (get_paths, load_blueprints, 
                                load_calibrations)

from pandas import read_json
from os import pardir, environ
from os.path import join, abspath, dirname

from numpy import arange


class Config(object):

    # Output level. 'QUIET': nothing, 'NORMAL': warn, err, 
    # 'DEBUG': info, warn, err
    out_level = 'NORMAL'

    # Usage
    framework = 'script'

    ### ---------------------------------------
    ### -----------------DATA------------------
    ### ---------------------------------------

    ## Place here options for data load and handling
    combined_devices_name = 'COMBINED_DEVICES'

    # Whether or not to reload smartcitizen firmware names from git repo
    reload_firmware_names = True

    # Whether or not to load cached data 
    # (saves time when requesting a lot of data)
    load_cached_api =  True

    # Whether or not to store loaded data 
    # (saves time when requesting a lot of data)
    store_cached_api = True

    # If reloading data from the API, how much gap between the saved data and the
    # latest reading in the API should be ignore
    cached_data_margin = 6  

    # If using multiple training datasets, how to call the joint df
    name_multiple_training_data = 'CDEV'

    ### ---------------------------------------
    ### -----------------PLOT------------------
    ### ---------------------------------------

    ### ---------------------------------------
    ### --------------ALGORITHMS---------------
    ### --------------------------------------- 

    # Whether or not to plot intermediate debugging visualisations in the algorithms
    intermediate_plots = False

    # Plot out level (priority of the plot to show - 'DEBUG' or 'NORMAL')
    plot_out_level = 'DEBUG'
    
    ### ---------------------------------------
    ### ----------------ZENODO-----------------
    ### ---------------------------------------

    # Urls
    zenodo_sandbox_base_url='http://sandbox.zenodo.org'
    zenodo_real_base_url='https://zenodo.org'
    
    ### ---------------------------------------
    ### -------------SMART CITIZEN-------------
    ### ---------------------------------------
    # Urls
    sensor_names_url_21='https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-21/master/lib/Sensors/Sensors.h'
    sensor_names_url_20='https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/master/lib/Sensors/Sensors.h'
    
    # Convertion table from API SC to Pandas
    # https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
    # https://developer.smartcitizen.me/#get-historical-readings
    freq_conv_lut = (
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
    alphadelta_pcb = 6.36
    # Deltas for baseline deltas algorithm
    baseline_deltas = arange(30, 45, 5)
    # Lambdas for baseline ALS algorithm
    baseline_als_lambdas = [1e5]
    
    ### ---------------------------------------
    ### -------------METRICS DATA--------------
    ### ---------------------------------------
    # Molecular weights of certain pollutants for unit convertion
    molecular_weights = {
                            'CO': 28, 
                            'NO': 30, 
                            'NO2': 46, 
                            'O3': 48,
                            'C6H6': 78,
                            'SO2': 64,
                            'H2S': 34
                        }
    # Background concentrations
    background_conc = {
                        'CO': 0, 
                        'NO2': 8, 
                        'O3': 40
                    }

    # Convolved metrics: metrics that measure two or more pollutants at once and that need to be
    # deconvolved
    convolved_metrics = ['NO2+O3']
    
    # This look-up table is comprised of channels you want always want to have with the same units and that might come from different sources
    # i.e. pollutant data in various units (ppm or ug/m3) from different analysers
    # The table should be used as follows:
    # 'key': 'units',
    # - 'key' is the channel that will lately be used in the analysis. It supports regex
    # - target_unit is the unit you want this channel to be and that will be converted in case of it being found in the channels list of your source
    
    channel_lut = {
                    "TEMP": "degC",
                    "HUM": "%rh",
                    "PRESS": "kPa",
                    "PM_(\d|[A,B]_\d)": "ug/m3",
                    "CO(\D|$)": "ppm",
                    "NOISE_A": "dBA",
                    "NO": "ppb",
                    "NO2": "ppb",
                    "NOX": "ppb",
                    "O3": "ppb",
                    "C6H6": "ppb",
                    "H2S": "ppb",
                    "SO2": "ppb"
                }

    # This table is used to convert units
    # ['from_unit', 'to_unit', 'multiplicative_factor', 'requires_M']
    # - 'from_unit'/'to_unit' = 'multiplicative_factor'
    # - 'requires_M' = whether it  
    # It accepts reverse operations - you don't need to put them twice but in reverse
    
    unit_convertion_lut = (
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
    plot_def_opt = {
                        'min_date': None,
                        'max_date': None,
                        'frequency': None,
                        'resample': 'mean',
                        'clean_na': 'fill',
                        'show': True
                    }

    map_def_opt = {
                    'location': [41.400818, 2.1825157],
                    'tiles': 'Stamen Toner',
                    'zoom': 2.5,
                    'period': '1W',
                    'radius': 10,
                    'fillOpacity': 1,
                    'stroke': 'false',
                    'icon': 'circle'
                    }

    plot_style = "seaborn-whitegrid"

    ts_plot_def_fmt = {
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
                            'style': plot_style
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
                            }
                      }

    scatter_plot_def_fmt = {
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
                            'style': plot_style,
                            'kind': 'reg',
                            'sharex': True,
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

    ts_scatter_def_fmt = {
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
                            'style': plot_style                         
                        }   
                    }

    heatmap_def_fmt = {
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
                            'style': plot_style,
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
    
    boxplot_def_fmt = {
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
                            'style': plot_style,
                            'frequency_hours': 2,
                            'session': '1D',
                            'palette': None, 
                            'periods': None
                            }
                      }

    dendrogram_def_fmt = {
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
    model_def_opt = {
                        'test_size': 0.2, 
                        'shuffle': False,
                        'clean_na': 'drop',
                        'common_avg': False
                    }

    model_hyperparameters = {
                                'rf': {

                                    'n_estimators': 100,
                                    'min_samples_leaf': 2,
                                    'max_features': None,
                                    'oob_score': True                         
                            }
                    }                        

    def __init__(self):
        self._is_init = False

    @property
    def is_init(self):
        return self._is_init
    
    def get_meta_data(self):
        if not self._is_init:
            paths = get_paths()

            if paths is not None: 
                blueprints = load_blueprints(paths)
                calibrations = load_calibrations(paths)

                self.paths = paths

                if calibrations is not None and blueprints is not None:
                    self.blueprints = blueprints
                    self.calibrations = calibrations

                    self._is_init = True
        
        return self.is_init

    def set_testing(self):
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
        self.out_level = 'DEBUG'
        self.framework = 'jupyterlab'
        self.intermediate_plots = True
        self.plot_out_level = 'DEBUG'