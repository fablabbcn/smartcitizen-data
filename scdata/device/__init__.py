''' Main implementation of class Device '''

from scdata.utils import std_out, localise_date, dict_fmerge, get_units_convf
from scdata.io import read_csv_file, export_csv_file
from scdata.utils import LazyCallable
from scdata._config import config
from scdata.device.process import *

from os.path import join
from pandas import DataFrame
from traceback import print_exc

class Device(object):
    ''' Main implementation of the device class '''

    def __init__(self, blueprint, descriptor):

        '''
        Creates an instance of device. Devices are objects that contain sensors readings, metrics 
        (calculations based on sensors readings), and metadata such as units, dates, frequency and source
        
        Parameters:
        -----------
            blueprint: String
            
            Defines the type of device. For instance: sck_21, sck_20, csic_station, muv_station
            parrot_soil, sc_20_station, sc_21_station. A list of all the blueprints is found in 
            blueprints.yaml and accessible via the scdata.utils.load_blueprints function.
            
            descriptor: dict()
            
            A dictionary containing information about the device itself. Depending on the blueprint, this descriptor
            needs to have different data. If not all the data is present, the corresponding blueprint's default will 
            be used
        Returns
        ----------
            Device object
        '''
        
        self.blueprint = blueprint

        # Set attributes
        for bpitem in config.blueprints[blueprint]: self.__setattr__(bpitem, config.blueprints[blueprint][bpitem]) 
        for ditem in descriptor.keys():
            if type(self.__getattribute__(ditem)) == dict: self.__setattr__(ditem, dict_fmerge(self.__getattribute__(ditem), descriptor[ditem]))
            else: self.__setattr__(ditem, descriptor[ditem])

        # Add API handler if needed
        if self.source == 'api':
            hmod = __import__('scdata.io.read_api', fromlist=['io.read_api'])
            Hclass = getattr(hmod, self.sources[self.source]['handler'])
            # Create object
            self.api_device = Hclass(did=self.id)

        self.readings = DataFrame()
        self.loaded = False
        self.options = dict()

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

    def load(self, options = None, path = None, convert_units = True):
        # Add test overrides if we have them, otherwise set device defaults
        if options is not None: self.check_overrides(options)
        else: self.check_overrides()

        try:
            if self.source == 'csv':
                self.readings = self.readings.combine_first(read_csv_file(join(path, self.processed_data_file), self.location, self.options['frequency'], 
                                                            self.options['clean_na'], self.sources[self.source]['index']))
                if self.readings is not None:
                    self.__convert_names__()

            elif 'api' in self.source:
                if path is None:
                    df = self.api_device.get_device_data(self.options['min_date'], self.options['max_date'], self.options['frequency'], self.options['clean_na'])
                    # API Device is not aware of other csv index data, so make it here
                    if 'csv' in self.sources and df is not None: df = df.reindex(df.index.rename(self.sources['csv']['index']))
                    # Combine it with readings if possible
                    if df is not None: self.readings = self.readings.combine_first(df)
                else:
                    # Cached case
                    self.readings = self.readings.combine_first(read_csv_file(join(path, str(self.id) + '.csv'), self.location, self.options['frequency'], 
                                                            self.options['clean_na'], self.sources['csv']['index']))
        except FileNotFoundError:
            std_out('File not found', 'ERROR')
            self.loaded = False
        except:
            print_exc()
            self.loaded = False
        else:
            if self.readings is not None: 
                self.__check_sensors__()
                if not self.readings.empty: 
                    self.loaded = True
                    if convert_units: self.__convert_units__()
        finally:
            return self.loaded

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

    def process(self, only_new = False, metrics = None):
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
        metrics: list
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

        if metrics is None: metrics = self.metrics

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

    # TODO
    # def capture(self):
    #     std_out('Not yet', 'ERROR')
