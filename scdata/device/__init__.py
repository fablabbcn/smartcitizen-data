''' Main implementation of class Device '''

from scdata.utils import std_out, localise_date, dict_fmerge, get_units_convf
from scdata.io import read_csv_file, export_csv_file
from scdata.utils import LazyCallable, url_checker, get_json_from_url
from scdata._config import config
from scdata.device.process import *

from os.path import join, basename
from urllib.parse import urlparse
from pandas import DataFrame, to_timedelta
from traceback import print_exc
from numpy import nan

class Device(object):
    ''' Main implementation of the device class '''

    def __init__(self, blueprint = None, descriptor = {}):

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

        self.skip_blueprint = False

        if blueprint is not None:
            self.blueprint = blueprint
            self.skip_blueprint = True
        else:
            self.blueprint = 'sck_21'

        # Set attributes
        if self.blueprint not in config.blueprints:
            raise ValueError(f'Specified blueprint {self.blueprint} is not in config')

        self.set_blueprint_attrs(config.blueprints[self.blueprint])
        self.blueprint_loaded_from_url = False
        self.hardware_loaded_from_url = False

        self.description = descriptor
        self.set_descriptor_attrs()

        if self.id is not None: self.id = str(self.id)

        # Postprocessing and forwarding
        self.hardware_url = None
        self.blueprint_url = None
        self.forwarding_params = None
        self.forwarding_request = None
        self.meta = None
        self.latest_postprocessing = None
        self.processed = False
        self.hardware_description = None

        # Add API handler if needed
        if self.source == 'api':

            hmod = __import__('scdata.io.device_api', fromlist = ['io.device_api'])
            Hclass = getattr(hmod, self.sources[self.source]['handler'])

            # Create object
            self.api_device = Hclass(did = self.id)

            std_out(f'Checking postprocessing info from API device')

            if self.load_postprocessing() and (self.hardware_url is None):# or self.blueprint_url is None):
                if config._strict:
                    raise ValueError('Postprocessing could not be loaded as is incomplete and strict mode is enabled')
                std_out(f'Postprocessing loaded but with problems (hardware_url: {self.hardware_url} // blueprint_url: {self.blueprint_url}', 'WARNING')

        if self.blueprint is None:
            raise ValueError(f'Device {self.id} cannot be init without blueprint. Need a blueprint to proceed')
        else:
            std_out(f'Device {self.id} is using {self.blueprint} blueprint')

        self.readings = DataFrame()
        self.loaded = False
        self.options = dict()
        std_out(f'Device {self.id} initialised', 'SUCCESS')

    def set_blueprint_attrs(self, blueprintd):

        # Set attributes
        for bpitem in blueprintd:
            self.__setattr__(bpitem, blueprintd[bpitem])

    def set_descriptor_attrs(self):

        # Descriptor attributes
        for ditem in self.description.keys():
            if ditem not in vars(self): std_out (f'Ignoring {ditem} from input'); continue
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

    def load_postprocessing(self):

        if self.source != 'api': return None

        if self.sources[self.source]['handler'] != 'ScApiDevice': return None

        # Request to get postprocessing information
        if self.api_device.get_device_postprocessing() is None: return None

        # Put it where it goes
        try:
            self.hardware_url = self.api_device.postprocessing['hardware_url']
            self.blueprint_url = self.api_device.postprocessing['blueprint_url']
            self.latest_postprocessing = self.api_device.postprocessing['latest_postprocessing']
            self.forwarding_params = self.api_device.postprocessing['forwarding_params']
            self.meta = self.api_device.postprocessing['meta']
            inc_postprocessing = False
        except KeyError:
            std_out('Ignoring postprocessing info as its incomplete', 'WARNING')
            inc_postprocessing = True
            pass

        if inc_postprocessing: return None

        # Load postprocessing info from url
        if url_checker(self.hardware_url) and self.hardware_loaded_from_url == False:

            std_out(f'Loading hardware information from:\n{self.hardware_url}')
            hardware_description = get_json_from_url(self.hardware_url)

            # TODO
            # Add additional checks to hardware_description

            if hardware_description is not None:
                self.hardware_description = hardware_description
                std_out('Hardware described in url is valid', "SUCCESS")
                self.hardware_loaded_from_url = True
            else:
                std_out("Hardware in url is not valid", 'ERROR')
                self.hardware_description = None

        # Find forwarding request
        if self.hardware_description is not None:
            if 'forwarding' in self.hardware_description:
                if self.hardware_description['forwarding'] in config.connectors:
                    self.forwarding_request = self.hardware_description['forwarding']
                    std_out(f"Requested a {self.hardware_description['forwarding']} connector for {self.id}")
                    if self.forwarding_params is None:
                        std_out('Assuming device has never been posted. Forwarding parameters are empty', 'WARNING')
                    else:
                        std_out(f'Connector parameters are not empty: {self.forwarding_params}')
                else:
                    std_out(f"Requested a {self.hardware_description['forwarding']} connector that is not available. Ignoring", 'WARNING')

        # Find postprocessing blueprint
        if self.skip_blueprint: std_out('Skipping blueprint as it was defined in device constructor', 'WARNING')
        if self.blueprint_loaded_from_url == False and not self.skip_blueprint:

            # Case when there is no info stored
            if url_checker(self.blueprint_url):
                std_out(f'blueprint_url in platform is not empty. Loading postprocessing blueprint from:\n{self.blueprint_url}')
                nblueprint = basename(urlparse(self.blueprint_url).path).split('.')[0]
            else:
                std_out(f'blueprint_url in platform is not valid', 'WARNING')
                std_out(f'Checking if there is a blueprint_url in hardware_description')
                if self.hardware_description is None:
                    std_out("Hardware description is not useful for blueprint", 'ERROR')
                    return None
                if 'blueprint_url' in self.hardware_description:
                    std_out(f"Trying postprocessing blueprint from:\n{self.hardware_description['blueprint_url']}")
                    nblueprint = basename(urlparse(self.hardware_description['blueprint_url']).path).split('.')[0]
                    tentative_urls = url_checker(self.hardware_description['blueprint_url'])
                    if len(tentative_urls)>0:
                        self.blueprint_url = tentative_urls[0]
                    else:
                        std_out('Invalid blueprint', 'ERROR')
                        return None
                else:
                    std_out('Postprocessing not possible without blueprint', 'ERROR')
                    return None

            std_out(f'Using hardware postprocessing blueprint: {nblueprint}')
            lblueprint = get_json_from_url(self.blueprint_url)

            if lblueprint is not None:
                self.blueprint = nblueprint
                self.blueprint_loaded_from_url = True
                self.set_blueprint_attrs(lblueprint)
                self.set_descriptor_attrs()
                std_out('Blueprint loaded from url', 'SUCCESS')
            else:
                std_out('Blueprint in url is not valid', 'ERROR')
                return None

        return self.api_device.postprocessing

    def validate(self):
        if self.hardware_description is not None: return True
        else: return False

    def load(self, options = None, path = None, convert_units = True, only_unprocessed = False, max_amount = None):
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
        max_amount: int
            Default: None
            Trim dataframe to this amount for processing and forwarding purposes
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
                self.location = self.api_device.get_device_timezone()

                if path is None:
                    # Not chached case
                    if only_unprocessed:

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
                if max_amount is not None:
                    std_out(f'Trimming dataframe to {max_amount} rows')
                    self.readings=self.readings.dropna(axis = 0, how='all').head(max_amount)
                if not self.readings.empty:
                    # Only add metrics if there is something that can be potentially processed
                    self.__fill_metrics__()
                    self.loaded = True
                    if convert_units: self.__convert_units__()
                else:
                    std_out('Empty dataframe in readings', 'WARNING')

        finally:
            self.processed = False
            return self.loaded

    def __fill_metrics__(self):
        std_out('Checking if metrics need to be added based on hardware info')

        if self.hardware_description is None:
            std_out(f'No hardware url in device {self.id}, ignoring')
            return None

        # Now go through sensor versions and add them to the metrics
        if 'versions' in self.hardware_description:
            for version in self.hardware_description['versions']:

                from_date = version["from"]
                to_date = version["to"]

                # Do not add any metric if the from_date of the calibration is after the last_reading_at
                # as there would be nothing to process
                if from_date > self.api_device.last_reading_at:
                    std_out('Postprocessing from_date is later than device last_reading_at', 'ERROR')
                    return None

                for slot in version["ids"]:

                    # Alphasense type - AAN 803-04
                    if slot.startswith('AS'):

                        sensor_id = version["ids"][slot]
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

                    # Other metric types will go here
            else:
                std_out('No hardware versions found, ignoring additional metrics', 'WARNING')

    def __check_sensors__(self):
        remove_sensors = list()
        # Remove sensor from the list if it's not in self.readings.columns
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
                # print_exc()
                std_out('Metric args not in dataframe', 'ERROR')
                process_ok=False
                pass

            if metric in self.readings: process_ok &= True

        if process_ok:
            # Latest postprocessing to latest readings
            if self.source == 'api':
                if self.api_device.get_device_postprocessing() is not None:
                    std_out('Updating postprocessing')
                    # Add latest postprocessing rounded up with frequency so that we don't end up in
                    # and endless loop processing only the latest data line (minute vs. second precission of the readings)
                    latest_postprocessing = localise_date(self.readings.index[-1]+to_timedelta(self.options['frequency']), 'UTC').strftime('%Y-%m-%dT%H:%M:%S')
                    self.api_device.postprocessing['latest_postprocessing'] = latest_postprocessing

                    std_out(f"{self.api_device.postprocessing}")
            std_out(f"Device {self.id} processed", "SUCCESS")

        self.processed = process_ok

        return process_ok

    def forward(self, chunk_size = 500, dry_run = False):
        '''
            Forwards data to another api
                Parameters
                ----------
                chunk_size: int
                    500
                    Chunk size to be sent to device.post_data_to_device in question
                dry_run: boolean
                    False
                    Post the payload to the API or just return it
            Returns
            ----------
                boolean
                True if posted ok, False otherwise
        '''

        if self.forwarding_params is None:
            std_out('Empty forwarding information', 'ERROR')
            return False

        rd = dict()
        df = self.readings.copy().dropna(axis = 0, how='all')

        df.rename(columns=rd, inplace=True)

        if df.empty:
            std_out('Empty dataframe, ignoring', 'WARNING')
            return False

        # Import requested handler
        hmod = __import__('scdata.io.device_api', fromlist = ['io.device_api'])
        Hclass = getattr(hmod, config.connectors[self.forwarding_request]['handler'])

        # Create object
        device = Hclass(did = self.forwarding_params)
        post_ok = device.post_data_to_device(df, chunk_size = chunk_size, dry_run = dry_run)
        if post_ok: std_out(f'Posted data for {self.id}', 'SUCCESS')
        else: std_out(f'Error posting data for {self.id}', 'ERROR')

        return post_ok

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

    def post_sensors(self, dry_run = False):
        '''
        Posts devices sensors. Only available for parent of ScApiDevice
            Parameters
            ----------
            dry_run: boolean
                False
                Post the payload to the API or just return it
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = True
        if self.sources[self.source]['handler'] != 'ScApiDevice':
            std_out('Only supported processing post is to SmartCitizen API', 'ERROR')
            return False

        rd = dict()
        df = self.readings.copy().dropna(axis = 0, how='all')
        for col in self.readings:
            if col not in self.sensors:
                std_out(f'Column ({col}) not in recognised IDs. Ignoring', 'WARNING')
                df.drop(col, axis=1, inplace=True)
                continue
            rd[col] = self.sensors[col]['id']

        df.rename(columns=rd, inplace=True)

        if df.empty:
            std_out('Empty dataframe, ignoring', 'WARNING')
            return False

        post_ok = self.api_device.post_data_to_device(df, dry_run = dry_run)
        if post_ok: std_out(f'Posted data for {self.id}', 'SUCCESS')
        else: std_out(f'Error posting data for {self.id}', 'ERROR')

        return post_ok

    def update_postprocessing(self, dry_run = False):
        '''
        Posts device postprocessing. Only available for parent of ScApiDevice
            Parameters
            ----------
            dry_run: boolean
                False
                Post the payload to the API or just return it
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = self.api_device.patch_postprocessing(dry_run=dry_run)

        if post_ok: std_out(f"Postprocessing posted for device {self.id}", "SUCCESS")
        return post_ok

    def post_metrics(self, with_postprocessing = False, dry_run = False):
        '''
        Posts devices metrics. Only available for parent of ScApiDevice
        Parameters
        ----------
            with_postprocessing: boolean
                False
                Post the postprocessing_attributes too
            dry_run: boolean
                False
                Post the payload to the API or just return it
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = True
        if self.sources[self.source]['handler'] != 'ScApiDevice':
            std_out('Only supported processing post is to SmartCitizen API', 'ERROR')
            return False

        rd = dict()
        std_out(f"Posting metrics for device {self.id}")
        # Make a copy of df
        df = self.readings.copy().dropna(axis = 0, how='all')
        # Get metrics to post, only the ones that have True in 'post' field and a valid ID
        # Replace their name with the ID to post
        for metric in self.metrics:
            if self.metrics[metric]['post'] == True:
                std_out(f"Adding {metric} for device {self.id}")
                rd[metric] = self.metrics[metric]['id']

        # Keep only metrics in df
        df = df[df.columns.intersection(list(rd.keys()))]
        df.rename(columns=rd, inplace=True)
        # Fill None or other values with actual NaN
        df = df.fillna(value=nan)

        # If empty, avoid
        if df.empty:
            std_out('Empty dataframe, ignoring', 'WARNING')
            return False

        post_ok = self.api_device.post_data_to_device(df, dry_run = dry_run)
        if post_ok: std_out(f'Posted metrics for {self.id}', 'SUCCESS')
        else: std_out(f'Error posting metrics for {self.id}', 'ERROR')

        # Post info if requested. It should be updated elsewhere
        if with_postprocessing and post_ok:
            post_ok &= self.update_postprocessing(dry_run=dry_run)

        if post_ok: std_out(f"Metrics posted for device {self.id}", "SUCCESS")
        return post_ok
