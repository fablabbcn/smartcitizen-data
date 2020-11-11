''' Main implementation of class Device '''

from scdata.utils import std_out, localise_date, dict_fmerge, get_units_convf
from scdata.io import read_csv_file, export_csv_file
from scdata.utils import LazyCallable
from scdata._config import config
from scdata.device.process import *

from os.path import join
from pandas import DataFrame
from traceback import print_exc
import datetime

class Device(object):
    ''' Main implementation of the device class '''

    def __init__(self, blueprint = 'sck_21', descriptor = {}):

        '''
        Creates an instance of device. Devices are objects that contain sensors readings, metrics 
        (calculations based on sensors readings), and metadata such as units, dates, frequency and source

        Parameters:
        -----------
        blueprint: String
            Default: 'sck_21'
            Defines the type of device. For instance: sck_21, sck_20, csic_station, muv_station
            parrot_soil, sc_20_station, sc_21_station. A list of all the blueprints is found in 
            blueprints.yaml and accessible via the scdata.utils.load_blueprints function.
            The blueprint can also be defined from the postprocessing info in SCAPI. The manual
            parameter passed here is prioritary to that of the API

        descriptor: dict()
            Default: empty dict
            A dictionary containing information about the device itself. Depending on the blueprint, this descriptor
            needs to have different data. If not all the data is present, the corresponding blueprint's default will 
            be used

        Examples:
        ----------
        Device('sck_21', descriptor = {'source': 'api', 'id': '1919'})
            device with sck_21 blueprint with 1919 ID
        Device(descriptor = {'source': 'api', 'id': '1919'})
            device with sck_21 blueprint with 1919 ID

        Returns
        ----------
            Device object
        '''

        if blueprint is not None:
            self.blueprint = blueprint

        # Set attributes
        for bpitem in config.blueprints[blueprint]: self.__setattr__(bpitem, config.blueprints[blueprint][bpitem])
        for ditem in descriptor.keys():
            if type(self.__getattribute__(ditem)) == dict: 
                self.__setattr__(ditem, dict_fmerge(self.__getattribute__(ditem), descriptor[ditem]))
            else: self.__setattr__(ditem, descriptor[ditem])    

        # Add API handler if needed
        if self.source == 'api':
            hmod = __import__('scdata.io.device_api', fromlist = ['io.device_api'])
            Hclass = getattr(hmod, self.sources[self.source]['handler'])
            # Create object
            self.api_device = Hclass(did = self.id)

        std_out(f'Checking postprocessing info from API device')
        self.load_postprocessing_info()

        if self.blueprint is None:
            std_out('Need a blueprint to proceed', 'ERROR')
            return None
        else:
            std_out(f'Device {self.id} is using {blueprint}')

        self.readings = DataFrame()
        self.loaded = False
        self.options = dict()

        self.hw_id = None
        self.latest_postprocessing = None

    def check_overrides(self, options = {}):
        
        if 'min_date' in options.keys(): self.options['min_date'] = options['min_date']
        else: self.options['min_date'] = self.min_date

        if 'max_date' in options.keys(): self.options['max_date'] = options['max_date']
        else: self.options['max_date'] = self.max_date

        if 'clean_na' in options.keys(): self.options['clean_na'] = options['clean_na']
        else: self.options['clean_na'] = self.clean_na

        if 'frequency' in options.keys(): self.options['frequency'] = options['frequency']
        elif self.frequency is not None: self.options['frequency'] = self.frequency
        else: self.options['frequency'] = '1Min'

    def load_postprocessing_info(self):

        if self.source != 'api':
            return None

        if self.sources[self.source]['handler'] != 'ScApiDevice':
            return None

        # Request to get postprocessing information
        if self.api_device.get_postprocessing_info() is None:
            return None

        # Put it where it goes
        self.hw_id = self.api_device.postprocessing_info['hardware_id']
        self.hw_updated_at = self.api_device.postprocessing_info['updated_at']
        self.hw_post_blueprint = self.api_device.postprocessing_info['postprocessing_blueprint']
        self.latest_postprocessing = self.api_device.postprocessing_info['latest_postprocessing']

        # Use postprocessing info blueprint
        if self.hw_post_blueprint in config.blueprints.keys():
            std_out(f'Using hardware postprocessing blueprint: {self.hw_post_blueprint}')
            self.blueprint = self.hw_post_blueprint

        return self.api_device.postprocessing_info

    def load(self, options = None, path = None, convert_units = True):

        # Add test overrides if we have them, otherwise set device defaults
        if options is not None: self.check_overrides(options)
        else: self.check_overrides()

        try:
            if self.source == 'csv':
                self.readings = self.readings.combine_first(read_csv_file(join(path, self.processed_data_file), self.location,
                                                            self.options['frequency'], self.options['clean_na'],
                                                            self.sources[self.source]['index']))
                if self.readings is not None:
                    self.__convert_names__()

            elif 'api' in self.source:

                # Get device location
                self.location = self.api_device.get_device_location()

                if path is None:
                    if self.load_postprocessing_info():
                        # Override dates for post-processing
                        if self.latest_postprocessing is not None:
                            hw_latest_post = localise_date(self.latest_postprocessing, self.location)
                            # Override min processing date
                            self.options['min_date'] = hw_latest_post

                    df = self.api_device.get_device_data(self.options['min_date'], self.options['max_date'],
                                                         self.options['frequency'], self.options['clean_na'])

                    # API Device is not aware of other csv index data, so make it here
                    if 'csv' in self.sources and df is not None:
                        df = df.reindex(df.index.rename(self.sources['csv']['index']))

                    # Combine it with readings if possible
                    if df is not None: self.readings = self.readings.combine_first(df)

                else:
                    # Cached case
                    self.readings = self.readings.combine_first(read_csv_file(join(path, str(self.id) + '.csv'),
                                                                self.location, self.options['frequency'],
                                                                self.options['clean_na'], self.sources['csv']['index']))

        except FileNotFoundError:
            # Handle error
            if 'api' in self.source: std_out(f'No cached data file found for device {self.id} in {path}. Moving on', 'WARNING')
            elif 'csv' in self.source: std_out(f'File not found for device {self.id} in {path}', 'ERROR')
            
            self.loaded = False
        except:
            print_exc()
            self.loaded = False
        else:
            if self.readings is not None: 
                self.__check_sensors__()

                if self.load_postprocessing_info() is not None:
                    self.__fill_metrics__()

                if not self.readings.empty:
                    self.loaded = True
                    if convert_units: self.__convert_units__()
        finally:
            return self.loaded

    def __fill_metrics__(self):

            
        if self.hw_id in config.hardware_info:
            std_out('Hardware ID found in history', "SUCCESS")
            hw_info = config.hardware_info[self.hw_id]
        else:
            std_out(f"Hardware id: {self.hw_id} not found in hardware_info", 'ERROR')
            return False

        # Now go through sensor versions and parse them
        for version in hw_info.keys():

            from_date = hw_info[version]["from"]
            to_date = hw_info[version]["to"]

            for slot in hw_info[version]["ids"]:

                # Alphasense type
                if slot.startswith('AS'):

                    sensor_id = hw_info[version]["ids"][slot]
                    as_type = config._as_sensor_codes[sensor_id[0:3]]
                    pollutant = as_type[as_type.index('_')+1:]
                    platform_sensor_id = config._platform_sensor_ids[pollutant]
                    # TODO - USE POLLUTANT OR PLATFORM SENSOR ID?
                    process = 'alphasense_803_04'

                    wen = f"ADC_{slot.strip('AS_')[:slot.index('_')]}_{slot.strip('AS_')[slot.index('_')+1]}"
                    aen = f"ADC_{slot.strip('AS_')[:slot.index('_')]}_{slot.strip('AS_')[slot.index('_')+2]}"

                    # metric_name = f'{pollutant}_V{version}_S{list(hw_info[version]["ids"]).index(slot)}'
                    metric_name = f'{pollutant}'

                    metric = {metric_name:
                                        {
                                            'process': process,
                                            'desc': f'Calculation of {pollutant} based on AAN 803-04',
                                            'units': 'ppb', # always for alphasense sensors,
                                            'id': platform_sensor_id,
                                            'post': True,
                                            'kwargs':  {
                                                        'from_date': from_date,
                                                        'to_date': to_date,
                                                        'id': sensor_id,
                                                        'we': wen,
                                                        'ae': aen,
                                                        't': 'EXT_TEMP', # With external temperature?
                                                        'location': self.location
                                                        }
                                        }
                    }

                self.add_metric(metric)


    def __check_sensors__(self):
        remove_sensors = list()
        for sensor in self.sensors:
            if sensor not in self.readings.columns: remove_sensors.append(sensor)

        if remove_sensors != []: std_out(f'Removing sensors from device: {remove_sensors}', 'WARNING')
        for sensor_to_remove in remove_sensors: self.sensors.pop(sensor_to_remove, None)
        std_out(f'Device sensors after removal: {list(self.sensors.keys())}')

    def __convert_names__(self):
        rename = dict()
        for sensor in self.sensors: 
            if 'id' in self.sensors[sensor] and sensor in self.readings.columns: rename[self.sensors[sensor]['id']] = sensor
        self.readings.rename(columns=rename, inplace=True)

    def __convert_units__(self):
        '''
            Convert the units based on the UNIT_LUT and blueprint
            NB: what is read/written from/to the cache is not converted.
            The files are with original units, and then converted in the device only
            for the readings but never chached like so.
        '''
        std_out('Checking if units need to be converted')
        for sensor in self.sensors:

            factor = get_units_convf(sensor, from_units = self.sensors[sensor]['units'])
            
            if factor != 1:
                self.readings.rename(columns={sensor: sensor + '_RAW'}, inplace=True)
                self.readings.loc[:, sensor] = self.readings.loc[:, sensor + '_RAW']*factor

    def process(self, only_new = False, lmetrics = None):
        '''
        Processes devices metrics, either added by the blueprint definition
        or the addition using Device.add_metric(). See help(Device.add_metric) for
        more information about the definition of the metrics to be added

        Parameters
        ----------
        only_new: boolean
            False
            To process or not the existing channels in the Device.readings that are
            defined in Device.metrics
        lmetrics: list
            None
            List of metrics to process. If none, processes all
        Returns
        ----------
            boolean
            True if processed ok, False otherwise
        '''

        process_ok = True

        if 'metrics' not in vars(self):
            std_out(f'Device {self.id} has nothing to process. Skipping', 'WARNING')
            return process_ok

        std_out('---------------------------')
        std_out(f'Processing device {self.id}')

        if lmetrics is None: metrics = self.metrics
        else: metrics = dict([(key, self.metrics[key]) for key in lmetrics])

        for metric in metrics:
            std_out(f'Processing {metric}')

            if only_new and metric in self.readings: 
                std_out(f'Skipping. Already in device')
                continue

            # Check if the metric contains a custom from_list
            if 'from_list' in metrics[metric]:
                lazy_name = metrics[metric]['from_list']
            else:
                lazy_name = f"scdata.device.process.{metrics[metric]['process']}"

            try:
                funct = LazyCallable(lazy_name)
            except ModuleNotFoundError:
                print_exc()
                process_ok &= False
                std_out('Problem adding lazy callable to metrics list', 'ERROR')
                pass
                return False

            args, kwargs = list(), dict()
            if 'args' in metrics[metric]: args = metrics[metric]['args']
            if 'kwargs' in metrics[metric]: kwargs = metrics[metric]['kwargs']

            try:
                self.readings[metric] = funct(self.readings, *args, **kwargs)
            except KeyError:
                print_exc()
                std_out('Metric args not in dataframe', 'ERROR')
                pass

            if metric in self.readings: process_ok &= True

        if process_ok:
            # Latest postprocessing to latest readings
            if self.api_device.get_postprocessing_info() is not None:
                latest_postprocessing = localise_date(self.readings.index[-1], self.location).strftime('%Y-%m-%dT%H:%M:%S')
                self.api_device.postprocessing_info['latest_postprocessing'] = latest_postprocessing

        return process_ok

    def add_metric(self, metric = dict()):
        '''
        Add a metric to the device to be processed by a callable function
        Parameters
        ----------
            metric: dict
            Empty dict
            Description of the metric to be added. It only adds it to
            Device.metrics, but does not calculate anything yet. The metric dict needs 
            to follow the format:
                metric = {
                            'metric_name': {'process': <function_name>
                                            'args': <iterable>
                                            'kwargs': <**kwargs for @function_name>
                                            'from_list': <module to load function from>
                            }
                }
            
            The 'from_list' parameter is optional, and onle needed if the process is not 
            already available in scdata.device.process.

            For a list of available processes call help(scdata.device.process)

            Example:
            --------
                metric = {'NO2_CLEAN': {'process': 'clean_ts',
                                        'kwargs': {'name': pollutant,
                                                   'limits': [0, 350],
                                                    'window_size': 5}
                        }}
        Returns
        ----------
        True if added metric
        '''

        if 'metrics' not in vars(self): 
            std_out(f'Device {self.id} has no metrics yet. Adding')
            self.metrics = dict()

        try:
            metricn = next(iter(metric.keys()))
            self.metrics[metricn] = metric[metricn]
        except:
            print_exc()
            return False

        std_out(f'Metric {metric} added to metrics', 'SUCCESS')
        return True

    def del_metric(self, metricn = ''):
        if 'metrics' not in vars(self): return
        if metricn in self.metrics: self.metrics.pop(metricn, None)
        if metricn in self.readings.columns: self.readings.__delitem__(metricn)
        
        if metricn not in self.readings and metricn not in self.metrics:
            std_out(f'Metric {metricn} removed from metrics', 'SUCCESS')
            return True
        return False        

    def export(self, path, forced_overwrite = False, file_format = 'csv'):
        '''
        Exports Device.readings to file
        Parameters
        ----------
            path: String
                Path to export file to, does not include filename.
                The filename will be the Device.id property
            forced_overwrite: boolean
                False
                Force data export in case of already existing file
            file_format: String
                'csv'
                File format to export. Current supported format CSV
        Returns
        ---------
            True if exported ok, False otherwise
        '''
        # Export device
        if file_format == 'csv':
            return export_csv_file(path, str(self.id), self.readings, forced_overwrite = forced_overwrite)
        else:
            std_out('Not supported format' ,'ERROR')
            return False

    def post(self, with_post_info = True):
        '''
        Posts devices metrics. Only available for parent of ScApiDevice
        Parameters
        ----------
            with_post_info: boolean
                Default True
                Add the post info to the package
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = True
        if self.sources[self.source]['handler'] != 'ScApiDevice':
            std_out('Only supported processing post is to SmartCitizen API', 'ERROR')
            return False

        for metric in self.metrics:
            if self.metrics[metric]['post'] == True:
                # Get single series for post
                df = DataFrame(self.readings[metric])
                sensor_id = self.metrics[metric]['id']
                post_ok &= self.api_device.post_device_data(df, sensor_id = sensor_id)

        # Post info if requested. It should be updated elsewhere
        if with_post_info: self.api_device.post_postprocessing_info()

        return post_ok

    # TODO
    # def capture(self):
    #     std_out('Not yet', 'ERROR')
