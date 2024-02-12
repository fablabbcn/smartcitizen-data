''' Main implementation of class Device '''

from scdata.utils import std_out, localise_date, dict_fmerge, get_units_convf
from scdata.io import read_csv_file, export_csv_file
from scdata.utils import LazyCallable, url_checker, get_json_from_url
from scdata._config import config
from scdata.device.process import *
from scdata.io.device_api import *

from os.path import join, basename
from urllib.parse import urlparse
from pandas import DataFrame, to_timedelta
from numpy import nan
from collections.abc import Iterable
from importlib import import_module

from timezonefinder import TimezoneFinder
tf = TimezoneFinder()

class Device(object):
    ''' Main implementation of the device class '''

    def __init__(self, blueprint = None, source=dict(), params=dict()):

        '''
        Creates an instance of device. Devices are objects that contain sensors data, metrics
        (calculations based on sensors data), and metadata such as units, dates, frequency and source

        Parameters:
        -----------
        blueprint: String
            Default: 'sck_21'
            Defines the type of device. For instance: sck_21, sck_20, csic_station, muv_station
            parrot_soil, sc_20_station, sc_21_station... A list of all the blueprints is found in
            config.blueprints_urls and accessible via the scdata.utils.load_blueprints(urls) function.
            The blueprint can also be defined from the postprocessing info in SCAPI.
            The manual parameter passed here overrides that of the API.

        source: dict()
            Default: empty dict
            A dictionary containing a description of how to obtain the data from the device itself.

        params: dict()
            Default: empty dict
            A dictionary containing information about the device itself. Depending on the blueprint, this params needs to have different data. If not all the data is present, the corresponding blueprint's default will be used

        Examples:
        ----------
        Device('sck_21', handler = {'source': 'api', 'id': '1919'}, params = {})
            device with sck_21 blueprint with 1919 ID

        Returns
        ----------
            Device object
        '''

        # Set handler
        self.source = source
        self.__set_handler__()

        # Set custom params
        self.params = params
        self.__set_blueprint_attrs__(config.blueprints['base'])
        self.__set_params_attrs__(params)

        # Start out handler object now
        if self.hclass is not None:
            self.handler = self.hclass(self.id)

        # Set blueprint
        if blueprint is not None:
            self.blueprint = blueprint
            if self.blueprint not in config.blueprints:
                raise ValueError(f'Specified blueprint {self.blueprint} is not in available blueprints')
            self.__set_blueprint_attrs__(config.blueprints[self.blueprint])
        else:
            if url_checker(self.handler.blueprint_url):
                std_out(f'Loading postprocessing blueprint from:\n{self.handler.blueprint_url}')
                self.blueprint = basename(urlparse(self.handler.blueprint_url).path).split('.')[0]
            else:
                raise ValueError(f'Specified blueprint url {self.handler.blueprint_url} is not valid')
            self.__set_blueprint_attrs__(self.handler.blueprint)

        # TODO Remove
        # self.__fill_handler_metrics__()

        # Init the rest of the stuff
        self.data = DataFrame()
        self.loaded = False
        self.processed = False
        self.postprocessing_updated = False
        std_out(f'Device {self.id} initialised', 'SUCCESS')

    def __set_handler__(self):
        # Add handlers here
        if self.source['type'] == 'api':
            if 'module' in self.source:
                module = self.source['module']
            else:
                module = 'scdata.io.device_api'
            try:
               hmod = import_module(self.source['module'])
            except ModuleNotFoundError:
                std_out(f"Module not found: {self.source['module']}")
                raise ModuleNotFoundError(f'Specified module not found')
            else:
                self.hclass = getattr(hmod, self.source['handler'])
                # Create object
                std_out(f'Setting handler as {self.hclass}')
        elif self.source['type'] == 'csv':
            # TODO Add handler here
            std_out ('No handler for CSV yet', 'ERROR')
            self.hclass = None
        elif self.source['type'] == 'kafka':
            # TODO Add handler here
            std_out ('No handler for kafka yet', 'ERROR')
            raise NotImplementedError

    def __set_blueprint_attrs__(self, blueprintd):

        # Set attributes
        for bpitem in blueprintd:
            if bpitem not in vars(self):
                self.__setattr__(bpitem, blueprintd[bpitem])
            elif self.__getattribute__(bpitem) is None:
                self.__setattr__(bpitem, blueprintd[bpitem])

    def __set_params_attrs__(self, params):

        # Params attributes
        for param in params.keys():
            if param not in vars(self):
                std_out (f'Ignoring {param} from input')
                continue
            if type(self.__getattribute__(param)) == dict:
                self.__setattr__(param, dict_fmerge(self.__getattribute__(param), params[param]))
            else:
                self.__setattr__(param, params[param])

    # TODO
    def validate(self):
        return True

    def merge_sensor_metrics(self, ignore_empty = True):
        std_out('Merging sensor and metrics channels')
        all_channels = dict_fmerge(self.sensors, self.metrics)

        if ignore_empty:
            to_ignore = []
            for channel in all_channels.keys():
                if channel not in self.data: to_ignore.append(channel)
                elif self.data[channel].dropna().empty:
                    std_out (f'{channel} is empty')
                    to_ignore.append(channel)

            [all_channels.pop(x) for x in to_ignore]

        return all_channels

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
        # TODO Except what?
        except:
            print_exc()
            return False

        std_out(f'Metric {metric} added to metrics', 'SUCCESS')
        return True

    def del_metric(self, metricn = ''):
        if 'metrics' not in vars(self): return
        if metricn in self.metrics: self.metrics.pop(metricn, None)
        if metricn in self.data.columns: self.data.__delitem__(metricn)

        if metricn not in self.data and metricn not in self.metrics:
            std_out(f'Metric {metricn} removed from metrics', 'SUCCESS')
            return True
        return False

    async def load(self, convert_units = True, only_unprocessed = False, max_amount = None, follow_defaults = False):
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
            Default: ''
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
        follow_defaults: bool
            Default: False
            Use defaults from config._csv_defaults for loading
        Returns
        ----------
            True if loaded correctly
        '''

        # # Add overrides if we have them, otherwise set device defaults
        # self.__check_overrides__(options)
        # std_out(f'Using options for device: {options}')

        if self.source['type'] == 'csv':
            # TODO Review if this is necessary
            if follow_defaults:
                index_name = config._csv_defaults['index_name']
                sep = config._csv_defaults['sep']
                skiprows = config._csv_defaults['skiprows']
            else:
                index_name = self.source['index_name']
                sep = self.source['sep']
                skiprows = self.source['skiprows']

            # TODO Change this for a csv handler
            # here we don't use tzaware because we only load preprocessed data
            try:
                csv_data = read_csv_file(
                        file_path = join(path, self.processed_data_file),
                        timezone = self.timezone,
                        frequency = self.frequency,
                        clean_na = self.clean_na,
                        resample = self.resample,
                        index_name = index_name,
                        sep = sep,
                        skiprows = skiprows)
            except FileNotFoundError:
                std_out(f'File not found for device {self.id} in {path}', 'ERROR')
            else:
                if csv_data is not None:
                    self.data = self.data.combine_first(csv_data)
                    self.__convert_names__()
                    self.loaded = self.__load_wrapup__(max_amount, convert_units)

        elif self.source['type'] == 'api':

            if self.handler.method == 'async':
                await self.handler.get_data(
                    min_date = self.min_date,
                    max_date = self.max_date,
                    freq = self.frequency,
                    clean_na = self.clean_na,
                    resample = self.resample,
                    only_unprocessed = only_unprocessed)
            else:
                self.handler.get_data(
                    min_date = self.min_date,
                    max_date = self.max_date,
                    freq = self.frequency,
                    clean_na = self.clean_na,
                    resample = self.resample,
                    only_unprocessed = only_unprocessed)
            # In principle this makes both dataframes as they are unmutable
            self.data = self.handler.data

            # Wrap it all up
            self.loaded = self.__load_wrapup__(max_amount, convert_units)

        elif self.source['type'] == 'kafka':
            std_out('Not yet', 'ERROR')
            raise NotImplementedError

        self.processed = False
        return self.loaded

    def __load_wrapup__(self, max_amount, convert_units):
        if self.data is not None:
            self.__check_sensors__()
            if not self.data.empty:
                if max_amount is not None:
                    # TODO Dirty workaround
                    std_out(f'Trimming dataframe to {max_amount} rows')
                    self.data=self.data.dropna(axis = 0, how='all').head(max_amount)
                # Convert units
                if convert_units:
                    self.__convert_units__()
                self.postprocessing_updated = False
                return True
            else:
                std_out('Empty dataframe in data', 'WARNING')
                return False
        else:
            return False

    # TODO remove
    def __fill_handler_metrics__(self):
        std_out('Checking if metrics need to be added based on hardware info')

        if self.handler.hardware_postprocessing is None:
            std_out(f'No hardware url in device {self.id}, ignoring')
            return None

        for metric in self.handler.metrics:
            metricn = next(iter(metric))
            if metricn not in self.metrics:
                std_out(f'Metric {metricn} from handler not in blueprint, ignoring.', 'WARNING')
                continue
            self.metrics[metricn]['kwargs'] = metric[metricn]['kwargs']

    def __check_sensors__(self):

        extra_sensors = list()
        # Check sensors from the list that are not in self.data.columns
        for sensor in self.sensors:
            if sensor not in self.data.columns:
                std_out(f'{sensor} not in data columns', 'INFO')
                extra_sensors.append(sensor)

        extra_columns = list()
        # Check columns from the data that are not in self.sensors
        for column in self.data.columns:
            if column not in self.sensors:
                extra_columns.append(column)
                std_out(f'Data contains extra columns: {extra_columns}', 'INFO')

        if config.data['strict_load']:
            std_out(f"config.data['strict_load'] is enabled. Removing extra columns")
            if extra_sensors != []:
                std_out(f'Removing sensors from device.sensors: {extra_sensors}', 'WARNING')
                for sensor_to_remove in extra_sensors:
                    self.sensors.pop(sensor_to_remove, None)
            if extra_columns != []:
                self.data.drop(extra_columns, axis=1, inplace=True)
        else:
            std_out(f"config.data['strict_load'] is disabled. Ignoring extra columns")

        std_out(f'Device sensors after checks: {list(self.sensors.keys())}')

    def __convert_names__(self):
        rename = dict()
        for sensor in self.sensors:
            if 'id' in self.sensors[sensor]:
                if self.sensors[sensor]['id'] in self.data.columns:
                    rename[self.sensors[sensor]['id']] = sensor
            else:
                std_out(f'No id in {self.sensors[sensor]}', 'WARNING')
        self.data.rename(columns=rename, inplace=True)

    def __convert_units__(self):
        '''
            Convert the units based on the UNIT_LUT and blueprint
            NB: what is read/written from/to the cache is not converted.
            The files are with original units, and then converted in the device only
            for the data but never chached like so.
        '''
        std_out('Checking if units need to be converted')
        for sensor in self.sensors:
            factor = get_units_convf(sensor, from_units = self.sensors[sensor]['units'])

            if factor != 1:
                self.data.rename(columns={sensor: sensor + '_in_' + self.sensors[sensor]['units']}, inplace=True)
                self.data.loc[:, sensor] = self.data.loc[:, sensor + '_in_' + self.sensors[sensor]['units']]*factor
        std_out('Units check done', 'SUCCESS')

    def process(self, only_new = False, lmetrics = None):
        '''
        Processes devices metrics, either added by the blueprint definition
        or the addition using Device.add_metric(). See help(Device.add_metric) for
        more information about the definition of the metrics to be added.

        Parameters
        ----------
        only_new: boolean
            False
            To process or not the existing channels in the Device.data that are
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
        self.postprocessing_updated = False

        if 'metrics' not in vars(self):
            std_out(f'Device {self.id} has nothing to process. Skipping', 'WARNING')
            return process_ok

        std_out('---------------------------')
        std_out(f'Processing device {self.id}')

        if lmetrics is None: metrics = self.metrics
        else: metrics = dict([(key, self.metrics[key]) for key in lmetrics])

        for metric in metrics:
            std_out(f'---')
            std_out(f'Processing {metric}')

            if only_new and metric in self.data:
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
                process_ok &= False
                std_out('Problem adding lazy callable to metrics list', 'ERROR')
                pass

            args, kwargs = list(), dict()
            if 'args' in metrics[metric]: args = metrics[metric]['args']
            if 'kwargs' in metrics[metric]: kwargs = metrics[metric]['kwargs']

            try:
                result = funct(self.data, *args, **kwargs)
            except KeyError:
                std_out('Cannot process requested function with data provided', 'ERROR')
                process_ok = False
                pass
            else:
                if result is not None:
                    self.data[metric] = result
                    process_ok &= True
                # If the metric is None, might be for many reasons and shouldn't collapse the process_ok

        if process_ok:
            std_out(f"Device {self.id} processed", "SUCCESS")
            self.processed = process_ok & self.update_postprocessing_date()

        return self.processed

    def update_postprocessing_date(self):

        latest_postprocessing = localise_date(self.data.index[-1]+\
            to_timedelta(self.frequency), 'UTC')
        if self.handler.update_latest_postprocessing(latest_postprocessing):
            if latest_postprocessing.to_pydatetime() == self.handler.latest_postprocessing:
                self.postprocessing_updated = True
            else:
                self.postprocessing_updated = False
        return self.postprocessing_updated

    # TODO
    def checks(self, level):
        '''
            Device checks
        '''
        # TODO Make checks dependent on each handler
        if self.source == 'api':
            # TODO normalise the functions accross all handlers
            # Check status code from curl
            response = self.api_device.checks()
            response['status'] = 200

        return response

    # TODO Remove
    def forward(self, chunk_size = 500, dry_run = False, max_retries = 2):
        '''
        Forwards data to another api.
        Parameters
        ----------
        chunk_size: int
            500
            Chunk size to be sent to device.post_data_to_device in question
        dry_run: boolean
            False
            Post the payload to the API or just return it
        max_retries: int
            2
            Maximum number of retries per chunk
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        # Import requested handler
        hmod = __import__('scdata.io.device_api', fromlist = ['io.device_api'])
        Hclass = getattr(hmod, config.connectors[self.forwarding_request]['handler'])

        # Create new device in target API if it hasn't been created yet
        if self.forwarding_params is None:
            std_out('Empty forwarding information, attemping creating a new device', 'WARNING')
            # We assume the device has never been posted
            # Construct new device kwargs dictionary
            kwargs = dict()
            for item in config.connectors[self.forwarding_request]['kwargs']:
                val = config.connectors[self.forwarding_request]['kwargs'][item]
                if val == 'options':
                    kitem = self.options[item]
                elif val == 'config':
                    # Items in config should be underscored
                    kitem = config.__getattr__(f'_{item}')
                elif isinstance(val, Iterable):
                    if 'same' in val:
                        if 'as_device' in val:
                            if item == 'sensors':
                                kitem = self.merge_sensor_metrics(ignore_empty = True)
                            elif item == 'description':
                                kitem = self.blueprint.replace('_', ' ')
                        elif 'as_api' in val:
                            if item == 'sensors':
                                kitem = self.api_device.get_device_sensors()
                            elif item == 'description':
                                kitem = self.api_device.get_device_description()
                else:
                    kitem = val
                kwargs[item] = kitem

            response = Hclass.new_device(name = config.connectors[self.forwarding_request]['name_prepend']\
                                                + str(self.id),
                                         location = self.location,
                                         dry_run = dry_run,
                                         **kwargs)
            if response:
                if 'message' in response:
                    if response['message'] == 'Created':
                        if 'sensorid' in response:
                            self.forwarding_params = response['sensorid']
                            self.api_device.postprocessing['forwarding_params'] = self.forwarding_params
                            std_out(f'New sensor ID in {self.forwarding_request}\
                             is {self.forwarding_params}. Updating')

        if self.forwarding_params is not None:
            df = self.data.copy()
            df = df[df.columns.intersection(list(self.merge_sensor_metrics(ignore_empty=True).keys()))]
            df = clean(df, 'drop', how = 'all')

            if df.empty:
                std_out('Empty dataframe, ignoring', 'WARNING')
                return False

            # Create object
            ndev = Hclass(did = self.forwarding_params)
            post_ok = ndev.post_data_to_device(df, chunk_size = chunk_size,
                dry_run = dry_run, max_retries = 2)

            if post_ok:
                # TODO Check if we like this
                if self.source == 'api':
                    self.update_latest_postprocessing()
                std_out(f'Posted data for {self.id}', 'SUCCESS')
            else:
                std_out(f'Error posting data for {self.id}', 'ERROR')
            return post_ok

        else:
            std_out('Empty forwarding information', 'ERROR')
            return False

    def export(self, path, forced_overwrite = False, file_format = 'csv'):
        '''
        Exports Device.data to file
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
            return export_csv_file(path, str(self.id), self.data, forced_overwrite = forced_overwrite)
        else:
            std_out('Not supported format' ,'ERROR')
            return False

    # TODO Check
    def post_sensors(self, clean_na = 'drop', chunk_size = 500, dry_run = False, max_retries = 2):
        '''
        Posts devices sensors. Only available for parent of ScApiDevice
            Parameters
            ----------
            clean_na: string, optional
                'drop'
                'drop', 'fill'
            chunk_size: integer
                chunk size to split resulting pandas DataFrame for posting data
            dry_run: boolean
                False
                Post the payload to the API or just return it
            max_retries: int
                2
                Maximum number of retries per chunk
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = True

        rd = dict()
        df = self.data.copy().dropna(axis = 0, how='all')
        for col in self.data:
            if col not in self.sensors:
                std_out(f'Column ({col}) not in recognised IDs. Ignoring', 'WARNING')
                df.drop(col, axis=1, inplace=True)
                continue
            rd[col] = self.sensors[col]['id']

        df.rename(columns=rd, inplace=True)

        if df.empty:
            std_out('Empty dataframe, ignoring', 'WARNING')
            return False

        std_out(f'Trying to post {list(df.columns)}')
        post_ok = self.handler.post_data_to_device(df, clean_na = clean_na,
            chunk_size = chunk_size, dry_run = dry_run, max_retries = max_retries)
        if post_ok: std_out(f'Posted data for {self.id}', 'SUCCESS')
        else: std_out(f'Error posting data for {self.id}', 'ERROR')

        return post_ok

    # TODO Check
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
        if not self.postprocessing_updated:
            std_out(f'Postprocessing is not up to date', 'ERROR')
            return False

        post_ok = self.api_device.patch_postprocessing(dry_run=dry_run)

        if post_ok: std_out(f"Postprocessing posted for device {self.id}", "SUCCESS")
        return post_ok

    # TODO Check
    def post_metrics(self, with_postprocessing = False, chunk_size = 500, dry_run = False, max_retries = 2):
        '''
        Posts devices metrics. Only available for parent of ScApiDevice
        Parameters
        ----------
            with_postprocessing: boolean
                False
                Post the postprocessing_attributes too
            chunk_size: integer
                chunk size to split resulting pandas DataFrame for posting data
            dry_run: boolean
                False
                Post the payload to the API or just return it
            max_retries: int
                2
                Maximum number of retries per chunk
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
        df = self.data.copy().dropna(axis = 0, how='all')
        # Get metrics to post, only the ones that have True in 'post' field and a valid ID
        # Replace their name with the ID to post
        for metric in self.metrics:
            if self.metrics[metric]['post'] == True and metric in self.data.columns:
                std_out(f"Adding {metric} for device {self.id} (ID: {self.metrics[metric]['id']})")
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

        std_out(f'Trying to post {list(df.columns)}')
        post_ok = self.api_device.post_data_to_device(df, chunk_size = chunk_size, dry_run = dry_run, max_retries = max_retries)
        if post_ok: std_out(f'Posted metrics for {self.id}', 'SUCCESS')
        else: std_out(f'Error posting metrics for {self.id}', 'ERROR')

        # Post info if requested. It should be updated elsewhere
        if with_postprocessing and post_ok:
            post_ok &= self.update_postprocessing(dry_run=dry_run)

        if post_ok: std_out(f"Metrics posted for device {self.id}", "SUCCESS")
        return post_ok
