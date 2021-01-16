''' Main implementation of class Device '''

from scdata.utils import std_out, localise_date, dict_fmerge, get_units_convf
from scdata.io import read_csv_file, export_csv_file
from scdata.utils import LazyCallable
from scdata.utils.meta import get_json_from_url
from scdata._config import config
from scdata.device.process import *

from os.path import join, basename
from urllib.parse import urlparse
from pandas import DataFrame, to_datetime
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
            config.blueprints_urls and accessible via the scdata.utils.load_blueprints(urls) function.
            The blueprint can also be defined from the postprocessing info in SCAPI. The manual
            parameter passed here is overriden by that of the API.

        descriptor: dict()
            Default: empty: std_out('Empty dataframe, ignoring', 'WARNING') dict
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
            self.set_blueprint_attrs(config.blueprints[blueprint])
            self.blueprint_loaded_from_url = False
            self.hw_loaded_from_url = False


        self.description = descriptor
        self.set_descriptor_attrs()

        if self.id is not None: self.id = str(self.id)

        # Add API handler if needed
        if self.source == 'api':

            hmod = __import__('scdata.io.device_api', fromlist = ['io.device_api'])
            Hclass = getattr(hmod, self.sources[self.source]['handler'])

            # Create object
            self.api_device = Hclass(did = self.id)

            std_out(f'Checking postprocessing info from API device')
            if self.load_postprocessing_info() is not None:
                std_out('Postprocessing info loaded successfully', 'SUCCESS')

        if self.blueprint is None:
            std_out('Need a blueprint to proceed', 'ERROR')
            return None
        else:
            std_out(f'Device {self.id} is using {self.blueprint}')

        self.readings = DataFrame()
        self.loaded = False
        self.options = dict()

        self.hw_url = None
        self.latest_postprocessing = None

    def set_blueprint_attrs(self, blueprintd):

        # Set attributes
        for bpitem in blueprintd:
            self.__setattr__(bpitem, blueprintd[bpitem])

    def set_descriptor_attrs(self):

        # Descriptor attributes
        for ditem in self.description.keys():
            if ditem not in vars(self): std_out (f'Ignoring {ditem} from input', 'WARNING'); continue
            if type(self.__getattribute__(ditem)) == dict:
                self.__setattr__(ditem, dict_fmerge(self.__getattribute__(ditem), self.description[ditem]))
            else: self.__setattr__(ditem, self.description[ditem])

    def check_overrides(self, options = {}):

        if 'min_date' in options.keys():
            self.options['min_date'] = options['min_date']
        else:
            self.options['min_date'] = self.min_date

        if 'max_date' in options.keys():
            self.options['max_date'] = options['max_date']
        else:
            self.options['max_date'] = self.max_date

        if 'clean_na' in options.keys():
            self.options['clean_na'] = options['clean_na']
        else:
            self.options['clean_na'] = self.clean_na

        if 'frequency' in options.keys():
            self.options['frequency'] = options['frequency']
        elif self.frequency is not None:
            self.options['frequency'] = self.frequency
        else:
            self.options['frequency'] = '1Min'

    def load_postprocessing_info(self):

        if self.source != 'api': return None

        if self.sources[self.source]['handler'] != 'ScApiDevice': return None

        # Request to get postprocessing information
        if self.api_device.get_postprocessing_info() is None: return None

        # Put it where it goes
        self.hw_url = self.api_device.postprocessing_info['hardware_url']
        self.hw_updated_at = self.api_device.postprocessing_info['updated_at']
        self.blueprint_url = self.api_device.postprocessing_info['blueprint_url']
        self.latest_postprocessing = self.api_device.postprocessing_info['latest_postprocessing']

        # Load hardware info from url
        if self.hw_url is not None and self.hw_loaded_from_url == False:

            std_out(f'Loading hardware info from:\n{self.hw_url}')
            lhw_info = get_json_from_url(self.hw_url)

            if lhw_info is not None:
                self.hw_info = lhw_info
                std_out('Hardware in url is valid', "SUCCESS")
                self.hw_loaded_from_url = True
            else:
                std_out("Hardware in url is not valid", 'ERROR')
                return None

        # Use postprocessing_info blueprint (not null case)
        if self.blueprint_url is not None and self.blueprint_loaded_from_url == False:

            std_out(f'Loading hardware postprocessing blueprint from:\n{self.blueprint_url}')
            nblueprint = basename(urlparse(self.blueprint_url).path).split('.')[0]
            std_out(f'Using hardware postprocessing blueprint: {nblueprint}')

            if nblueprint in config.blueprints:

                std_out(f'Blueprint from hardware info ({nblueprint}) already in config.blueprints. Overwritting')
                # self.blueprint_loaded_from_url = True
                # return self.api_device.postprocessing_info

            lblueprint = get_json_from_url(self.blueprint_url)

            if lblueprint is not None:

                std_out('Blueprint loaded from url', 'SUCCESS')
                self.blueprint = nblueprint
                self.blueprint_loaded_from_url = True
                self.set_blueprint_attrs(lblueprint)
                self.set_descriptor_attrs()

            else:

                std_out('Blueprint in url is not valid', 'ERROR')
                return None

        # Use postprocessing_info blueprint (null case)
        elif self.blueprint_url is None and self.blueprint_loaded_from_url == False:

            if 'default_blueprint_url' in self.hw_info:

                std_out(f"Loading default hardware postprocessing blueprint from:\n{self.hw_info['default_blueprint_url']}")
                nblueprint = basename(urlparse(self.hw_info['default_blueprint_url']).path).split('.')[0]

                if nblueprint in config.blueprints:
                    std_out(f'Default blueprint from hardware info ({nblueprint}) already in config.blueprints. Overwritting', 'WARNING')

                lblueprint = get_json_from_url(self.hw_info['default_blueprint_url'])

                if lblueprint is not None:

                    std_out('Default lueprint loaded from url', 'SUCCESS')
                    self.blueprint = nblueprint
                    self.blueprint_loaded_from_url = True
                    self.set_blueprint_attrs(lblueprint)
                    self.set_descriptor_attrs()

                else:

                    std_out('Blueprint in url is not valid', 'ERROR')
                    return None

            else:

                std_out('Postprocessing not possible without blueprint', 'ERROR')
                return None

        return self.api_device.postprocessing_info

    def load(self, options = None, path = None, convert_units = True, only_unprocessed = False):
        '''
        Loads the device with some options

        Parameters:
        -----------
        options: dict()
            Default: None
            options['min_date'] = date to load data from
                Default to device min_date (from blueprint or test)
            options['max_date'] = date to load data to
                Default to device max_date (from blueprint or test)
            options['clean_na'] = clean na (drop_na, fill_na or None)
                Default to device clean_na (from blueprint or test)
            options['frequency'] = frequency to load data at in pandas format
                Default to device frequency (from blueprint or test) or '1Min'
        path: String
            Default: None
            Path were the csv file is, if any. Normally not needed to be provided, only for internal usage
        convert_units: bool
            Default: True
            Convert units for channels based on config._channel_lut
        only_unprocessed: bool
            Default: False
            Loads only unprocessed data

        Returns
        ----------
            True if loaded correctly
        '''

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

                    if self.load_postprocessing_info() and only_unprocessed:

                        # Override dates for post-processing
                        if self.latest_postprocessing is not None:
                            hw_latest_postprocess = localise_date(self.latest_postprocessing, 'UTC').strftime('%Y-%m-%dT%H:%M:%S')
                            # Override min loading date
                            self.options['min_date'] = hw_latest_postprocess

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
                self.__fill_metrics__()

                if not self.readings.empty:
                    self.loaded = True
                    if convert_units: self.__convert_units__()
        finally:
            return self.loaded

    def __fill_metrics__(self):
        std_out('Checking if metrics need to be added based on hardware info')

        if self.hw_url is None:
            std_out(f'No hardware url in device {self.id}, ignoring')
            return None

        # Now go through sensor versions and parse them
        for version in self.hw_info.keys():

            from_date = self.hw_info[version]["from"]
            to_date = self.hw_info[version]["to"]

            for slot in self.hw_info[version]["ids"]:

                # Alphasense type - AAN 803-04
                if slot.startswith('AS'):

                    sensor_id = self.hw_info[version]["ids"][slot]
                    as_type = config._as_sensor_codes[sensor_id[0:3]]
                    pollutant = as_type[as_type.index('_')+1:]
                    if pollutant == 'OX': pollutant = 'O3'

                    # Get working and auxiliary electrode names
                    wen = f"ADC_{slot.strip('AS_')[:slot.index('_')]}_{slot.strip('AS_')[slot.index('_')+1]}"
                    aen = f"ADC_{slot.strip('AS_')[:slot.index('_')]}_{slot.strip('AS_')[slot.index('_')+2]}"

                    if pollutant not in self.metrics:
                        # Create Metric
                        std_out(f'Metric {pollutant} not in blueprint, ignoring.', 'WARNING')
                    else:
                        # Simply fill it up
                        std_out(f'{pollutant} found in blueprint metrics, filling up with hardware info')
                        self.metrics[pollutant]['kwargs']['we'] = wen
                        self.metrics[pollutant]['kwargs']['ae'] = aen
                        self.metrics[pollutant]['kwargs']['location'] = self.location
                        self.metrics[pollutant]['kwargs']['alphasense_id'] = str(sensor_id)
                        self.metrics[pollutant]['kwargs']['from_date'] = from_date
                        self.metrics[pollutant]['kwargs']['to_date'] = to_date

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
        std_out('Units check done', 'SUCCESS')

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
                std_out('Updating postprocessing_info')
                latest_postprocessing = localise_date(self.readings.index[-1], 'UTC').strftime('%Y-%m-%dT%H:%M:%S')
                self.api_device.postprocessing_info['latest_postprocessing'] = latest_postprocessing
                self.api_device.postprocessing_info['updated_at'] = to_datetime(datetime.datetime.now(), utc = False)\
                                                                    .tz_localize(config._location)\
                                                                    .tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S')
                std_out(f"{self.api_device.postprocessing_info}")
                std_out(f"Device {self.id} processed", "SUCCESS")

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

    def post_sensors(self):
        '''
        Posts devices sensors. Only available for parent of ScApiDevice
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = True
        if self.sources[self.source]['handler'] != 'ScApiDevice':
            std_out('Only supported processing post is to SmartCitizen API', 'ERROR')
            return False

        for sensor in self.sensors:
            std_out(f'Posting sensor: {sensor}')
            # Get single series for post
            df = DataFrame(self.readings[sensor]).dropna(axis = 0, how='all')
            if df.empty: 
                std_out('Empty dataframe, ignoring', 'WARNING')
                continue
            sensor_id = self.sensors[sensor]['id']
            post_ok &= self.api_device.post_device_data(df, sensor_id = sensor_id)

        return post_ok
        
    def post_metrics(self, with_post_info = True):
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

        std_out(f"Posting metrics for device {self.id}")
        for metric in self.metrics:
            if self.metrics[metric]['post'] == True:
                std_out(f"Posting {metric} for device {self.id}")
                # Get single series for post
                df = DataFrame(self.readings[metric]).dropna(axis = 0, how='all')
                if df.empty:
                    std_out('Empty dataframe, ignoring', 'WARNING')
                    continue
                sensor_id = self.metrics[metric]['id']
                post_ok &= self.api_device.post_device_data(df, sensor_id = sensor_id)
                if post_ok: std_out(f"Metric {metric} posted", "SUCCESS")
                else: std_out(f"Error while posting {metric}", "WARNING")

        # Post info if requested. It should be updated elsewhere
        if with_post_info:
            post_ok &= self.api_device.post_postprocessing_info()

        if post_ok: std_out(f"Metrics posted for device {self.id}", "SUCCESS")
        return post_ok
