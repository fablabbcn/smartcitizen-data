''' Main implementation of class Device '''

from scdata.tools.custom_logger import logger
from scdata.io import read_csv_file, export_csv_file
from scdata.tools.lazy import LazyCallable
from scdata.tools.url_check import url_checker
from scdata.tools.date import localise_date
from scdata.tools.dictmerge import dict_fmerge
from scdata.tools.units import get_units_convf
from scdata.tools.find import find_by_field
from scdata._config import config
from scdata.io.device_api import *
from scdata.models import Blueprint, Metric, Source, APIParams, CSVParams, DeviceOptions, Sensor

from os.path import join, basename
from urllib.parse import urlparse
from pandas import DataFrame, to_timedelta, Timedelta
from numpy import nan
from collections.abc import Iterable
from importlib import import_module
from pydantic import TypeAdapter, BaseModel, ConfigDict
from pydantic_core import ValidationError
from typing import Optional, List
from json import dumps

from timezonefinder import TimezoneFinder
tf = TimezoneFinder()

class Device(BaseModel):
    ''' Main implementation of the device class '''
    model_config = ConfigDict(arbitrary_types_allowed = True)

    blueprint: str = None
    source: Source = Source()
    options: DeviceOptions = DeviceOptions()
    params: object = None
    paramsParsed: object = None
    metrics: List[Metric] = []
    meta: dict = dict()
    hclass: object = None
    handler: object = None
    data: DataFrame = DataFrame()
    loaded: bool = False
    processed: bool = False
    postprocessing_updated: bool = False

    def model_post_init(self, __context) -> None:

        '''
        Creates an instance of device. Devices are objects that contain sensors data, metrics
        (calculations based on sensors data), and metadata such as units, dates, frequency and source

        Parameters:
        -----------
        blueprint: String
            Default: 'sck_21'
            Defines the type of device. For instance: sck_21, sck_20, csic_station, muv_station
            parrot_soil, sc_20_station, sc_21_station... A list of all the blueprints is found in
            config.blueprints_urls and accessible via the scdata.tools.load_blueprints(urls) function.
            The blueprint can also be defined from the postprocessing info in SCAPI.
            The manual parameter passed here overrides that of the API.

        source: dict()
            Default: empty dict
            A dictionary containing a description of how to obtain the data from the device itself.

        params: dict()
            Default: empty dict
            A dictionary containing information about the device itself. Depending on the blueprint, this params needs to have different data. If not all the data is present, the corresponding blueprint's default will be used

        Returns
        ----------
            Device object
        '''

        # Set handler
        self.__set_handler__()
        # Set blueprint
        if self.blueprint is not None:
            if self.blueprint not in config.blueprints:
                raise ValueError(f'Specified blueprint {self.blueprint} is not in available blueprints')
            self.__set_blueprint_attrs__(config.blueprints[self.blueprint])
        else:
            if url_checker(self.handler.blueprint_url):
                logger.info(f'Loading postprocessing blueprint from:\n{self.handler.blueprint_url}')
                self.blueprint = basename(urlparse(self.handler.blueprint_url).path).split('.')[0]
                self.__set_blueprint_attrs__(self.handler.properties)
            else:
                raise ValueError(f'Specified blueprint url {self.handler.blueprint_url} is not valid')

        logger.info(f'Device {self.paramsParsed.id} initialised')

    def __set_handler__(self):
        # Add handlers here
        if self.source.type == 'api':
            try:
                module = self.source.module
            except:
                # Default to device_api if not specified
                module = 'scdata.io.device_api'
                logger.warning(f'Module not specified. Defaulting to {module}')
                pass

            # Try to find module
            try:
               hmod = import_module(module)
            except ModuleNotFoundError:
                logger.error(f"Module not found: {module}")
                raise ModuleNotFoundError(f'Specified module not found')
            else:
                self.hclass = getattr(hmod, self.source.handler)
                logger.info(f'Setting handler as {self.hclass}')

            self.paramsParsed = TypeAdapter(APIParams).validate_python(self.params)

        elif self.source.type == 'csv':
            try:
                module = self.source.module
            except:
                # Default to device_file if not specified
                module = 'scdata.io.device_file.CSVHandler'
                logger.warning(f'Module not specified. Defaulting to {module}')
                pass

            # Try to find module
            try:
               hmod = import_module(module)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f'Specified module not found {module}')
            else:
                self.hclass = getattr(hmod, self.source.handler)
                logger.info(f'Setting handler as {self.hclass}')

            self.paramsParsed = TypeAdapter(CSVParams).validate_python(self.params)
        elif self.source.type == 'stream':
            # TODO Add handler here
            raise NotImplementedError('No handler for stream yet')

        if self.hclass is not None:
            self.handler = self.hclass(params = self.paramsParsed)
        else:
            raise ValueError("Devices need one handler")

    def __set_blueprint_attrs__(self, blueprint):
        # Set attributes
        for item in blueprint:
            if item not in vars(self):
                raise ValueError(f'Invalid blueprint item {item}')
            else:
                # Workaround for postponed fields
                item_type = self.model_fields[item].annotation
                self.__setattr__(item, TypeAdapter(item_type).validate_python(blueprint[item]))

        # Sensors renaming dict
        self._sensors = TypeAdapter(List[Sensor]).validate_python(self.handler.sensors)

        self._rename = dict()
        for channel in self._sensors:
            if channel.id is not None:
                _ch = find_by_field(config.names[self.source.handler], channel.id, 'id')
                if _ch:
                    self._rename[channel.name] = _ch.name
            else:
                logger.warning(f'Channel {channel.name} has no id')

        # Metrics stay the same
        for channel in self.metrics:
            self._rename[channel.name] = channel.name

    # TODO - Improve?
    @property
    def valid_for_processing(self):
        if self.blueprint is not None and \
            self.handler.hardware_postprocessing is not None and \
            self.handler.postprocessing is not None and \
            self.handler.blueprint_url is not None:
            return True
        else:
            return False

    @property
    def id(self):
        return self.paramsParsed.id

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
                            'id': Optional[int] = None
                                <sensor_id> (in an API for instance)
                            'description': Optional[str] = ''
                                <description>
                            'name': [str]
                                <metric_name>
                            'function': [str]
                                <function_name>
                            'args': Optional[dict] = None
                                <iterable of arguments>
                            'kwargs': Optional[dict] = None
                                <**kwargs for @function_name>
                            'module': Optional[str] = 'scdata.device.process
                                <module to load function from>'
                        }
            The 'module' parameter is only needed if the process is not
            already available in scdata.device.process.
        Example:
        --------
        d.add_metric(metric={
                        'name': 'NO2_CLEAN',
                        'function': 'clean_ts',
                        'description': 'Clean NO2 channel',
                        'units': 'ppb',
                        'kwargs': {
                            'name': 'NO2',
                            'limits': [0, 350],
                            'window_size': 5}
                     })
        Returns
        ----------
            True if added metric
        '''

        if 'metrics' not in vars(self):
            logger.info(f'Device {self.paramsParsed.id} has no metrics yet. Adding')
            self.metrics = list()

        _metric = TypeAdapter(Metric).validate_python(metric)

        if self.__check_callable__(_metric.module, _metric.function):
            self.metrics.append(_metric)

            logger.info(f"Metric {_metric.name} added to metrics")
            self._rename[_metric.name] = _metric.name
            return True

    def del_metric(self, metric_name = ''):
        if 'metrics' not in vars(self): raise ValueError('Device has no metrics')
        m = find_by_field(self.metrics, metric_name, 'name')
        if m:
            self.metrics.remove(m)
        else:
            logger.warning(f'Metric {metric_name} not in metrics')
            return False
        if metric_name in self.data.columns:
            self.data.__delitem__(metric_name)

        if metric_name not in self.data and find_by_field(self.metrics, metric_name, 'name') is None:
            logger.info(f'Metric {metric_name} removed from metrics')
            return True
        return False

    async def load(self, cache=None, convert_units=True, convert_names=True, max_amount=None):
        '''
        Loads the device with some options

        Parameters:
        -----------
        cache: String
            Default: None
            Path were the cached file is, if any. Normally not needed to be provided, only for internal usage
        convert_units: bool
            Default: True
            Convert units for channels based on config._channel_lut
        convert_names: bool
            Default: True
            Convert names for channels based on ids
        max_amount: int
            Default: None
            Trim dataframe to this amount for processing and forwarding purposes (workaround)
        Returns
        ----------
            True if loaded correctly
        '''
        min_date = self.options.min_date
        max_date = self.options.max_date
        timezone = self.handler.timezone
        frequency = self.options.frequency
        clean_na = self.options.clean_na
        resample = self.options.resample
        cached_data = DataFrame()

        # Only case where cache makes sense
        if self.source.type == 'api':
            if cache is not None and cache:
                if cache.endswith('.csv'):
                    cached_data = read_csv_file(
                        path = cache,
                        timezone = timezone,
                        frequency = frequency,
                        clean_na = clean_na,
                        resample = resample,
                        index_name = 'TIME')
                else:
                    raise NotImplementedError(f'Cache needs to be a .csv file. Got {cache}.')

                if not cached_data.empty:
                    # Update min_date
                    min_date=cached_data.index[-1].tz_convert('UTC')+Timedelta(frequency)

        # Not implemented "for now"
        elif self.source.type == 'stream':
            raise NotImplementedError('Source type stream not implemented yet')

        # The methods below should be implemented from the handler type
        if self.handler.method == 'async':
            await self.handler.get_data(
                min_date = min_date,
                max_date = max_date,
                frequency = frequency,
                clean_na = clean_na,
                resample = resample)
        else:
            self.handler.get_data(
                min_date = min_date,
                max_date = max_date,
                frequency = frequency,
                clean_na = clean_na,
                resample = resample)

        # In principle this links both dataframes as they are unmutable
        self.data = self.handler.data
        # Wrap it all up
        self.loaded = self.__load_wrapup__(max_amount,  convert_units=convert_units, convert_names=convert_names, cached_data=cached_data)

        self.processed = False
        return self.loaded

    def __load_wrapup__(self, max_amount, convert_units=True, convert_names=True, cached_data=None):
        if self.data is not None:
            if not self.data.empty:
                if max_amount is not None:
                    # TODO Dirty workaround
                    logger.info(f'Trimming dataframe to {max_amount} rows')
                    self.data=self.data.dropna(axis = 0, how='all').head(max_amount)
                # Convert names
                if convert_names:
                    self.__convert_names__()
                # Convert units
                if convert_units:
                    self.__convert_units__()
                self.postprocessing_updated = False
            else:
                logger.info('Empty dataframe in loaded data. Waiting for cache...')

        if not cached_data.empty:
            logger.info('Cache exists')
            self.data = self.data.combine_first(cached_data)

        return not self.data.empty

    def __convert_names__(self):
        logger.info('Converting names...')

        self.data.rename(columns=self._rename, inplace=True)
        logger.info('Names converted')

    def __convert_units__(self):
        '''
            Convert the units based on the UNIT_LUT and blueprint
            NB: what is read/written from/to the cache is not converted.
            The files are with original units, and then converted in the device only
            for the data but never chached like so.
        '''
        logger.info('Checking if units need to be converted...')
        for sensor in self.data.columns:
            _rename_inv = {v: k for k, v in self._rename.items()}
            if sensor not in _rename_inv:
                logger.info(f'Sensor {sensor} not renamed. Units check not needed')
                continue
            sensorm = find_by_field(self._sensors, _rename_inv[sensor], 'name')

            if sensorm is not None:
                factor = get_units_convf(sensor, from_units = sensorm.unit)

                if factor != 1:
                    self.data.rename(columns={sensor: sensor + '_in_' + sensorm.unit}, inplace=True)
                    self.data.loc[:, sensor] = self.data.loc[:, sensor + '_in_' + sensorm.unit]*factor
                else:
                    logger.info(f'No units conversion needed for sensor {sensor} (factor=1)')
            else:
                logger.warning('Sensor not found')
        logger.info('Units check done')

    def __check_callable__(self, module, function):
        # Check if the metric contains a custom module
        lazy_name = f"{module}.{function}"

        try:
            funct = LazyCallable(lazy_name)
        except ModuleNotFoundError:
            logger.error(f'Callable {lazy_name} not available')
            return False
        else:
            return True
        return False

    def process(self, only_new=False, lmetrics=None):
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

        if not self.loaded:
            logger.error('Need to load first (device.load())')
            return False

        process_ok = True
        self.postprocessing_updated = False

        if 'metrics' not in vars(self):
            logger.warning(f'Device {self.paramsParsed.id} has nothing to process. Skipping')
            return process_ok

        logger.info('---------------------------')
        logger.info(f'Processing device {self.paramsParsed.id}')
        if lmetrics is None:
            _lmetrics = [metric.name for metric in self.metrics]
        else: _lmetrics = lmetrics

        if not _lmetrics:
            logger.warning('Nothing to process')
            return process_ok

        for metric in self.metrics:
            logger.info('---')
            if metric.name not in _lmetrics: continue
            logger.info(f'Processing {metric.name}')

            if only_new and metric.name in self.data:
                logger.info(f'Skipping. Already in device')
                continue

            if self.__check_callable__(metric.module, metric.function):
                funct = LazyCallable(f"{metric.module}.{metric.function}")
            else:
                process_ok &= False
                logger.error('Problem adding lazy callable to metrics list')
                continue

            args, kwargs = list(), dict()
            if 'args' in vars(metric):
                if metric.args is not None: args = metric.args
            if 'kwargs' in vars(metric):
                if metric.kwargs is not None: kwargs = metric.kwargs

            try:
                result = funct(self.data, *args, **kwargs)
            except KeyError:
                logger.error('Cannot process requested function with data provided')
                process_ok = False
                pass
            else:
                if result is not None:
                    self.data[metric.name] = result
                    process_ok &= True
                # If the metric is None, might be for many reasons and shouldn't collapse the process_ok

        if process_ok:
            logger.info(f"Device {self.paramsParsed.id} processed")
            self.processed = process_ok & self.update_postprocessing_date()

        return self.processed

    @property
    def sensors(self):
        return self._sensors

    def update_postprocessing_date(self):
        latest_postprocessing = localise_date(self.data.index[-1]+\
            to_timedelta(self.options.frequency), 'UTC')
        if self.handler.update_latest_postprocessing(latest_postprocessing):
            # Consider the case of no postprocessing, to avoid making the whole thing false
            if latest_postprocessing.to_pydatetime() == self.handler.latest_postprocessing or self.handler.json.postprocessing is None:
                self.postprocessing_updated = True
            else:
                self.postprocessing_updated = False
        return self.postprocessing_updated

    # TODO
    def health_check(self):
        return True

    # TODO - Decide if we keep it
    # def forward(self, chunk_size = 500, dry_run = False, max_retries = 2):
    #     '''
    #     Forwards data to another api.
    #     Parameters
    #     ----------
    #     chunk_size: int
    #         500
    #         Chunk size to be sent to device.post_data_to_device in question
    #     dry_run: boolean
    #         False
    #         Post the payload to the API or just return it
    #     max_retries: int
    #         2
    #         Maximum number of retries per chunk
    #     Returns
    #     ----------
    #         boolean
    #         True if posted ok, False otherwise
    #     '''

    #     # Import requested handler
    #     hmod = __import__('scdata.io.device_api', fromlist = ['io.device_api'])
    #     Hclass = getattr(hmod, config.connectors[self.forwarding_request]['handler'])

    #     # Create new device in target API if it hasn't been created yet
    #     if self.forwarding_params is None:
    #         std_out('Empty forwarding information, attemping creating a new device', 'WARNING')
    #         # We assume the device has never been posted
    #         # Construct new device kwargs dictionary
    #         kwargs = dict()
    #         for item in config.connectors[self.forwarding_request]['kwargs']:
    #             val = config.connectors[self.forwarding_request]['kwargs'][item]
    #             if val == 'options':
    #                 kitem = self.options[item]
    #             elif val == 'config':
    #                 # Items in config should be underscored
    #                 kitem = config.__getattr__(f'_{item}')
    #             elif isinstance(val, Iterable):
    #                 if 'same' in val:
    #                     if 'as_device' in val:
    #                         if item == 'sensors':
    #                             kitem = self.merge_sensor_metrics(ignore_empty = True)
    #                         elif item == 'description':
    #                             kitem = self.blueprint.replace('_', ' ')
    #                     elif 'as_api' in val:
    #                         if item == 'sensors':
    #                             kitem = self.api_device.get_device_sensors()
    #                         elif item == 'description':
    #                             kitem = self.api_device.get_device_description()
    #             else:
    #                 kitem = val
    #             kwargs[item] = kitem

    #         response = Hclass.new_device(name = config.connectors[self.forwarding_request]['name_prepend']\
    #                                             + str(self.params.id),
    #                                      location = self.location,
    #                                      dry_run = dry_run,
    #                                      **kwargs)
    #         if response:
    #             if 'message' in response:
    #                 if response['message'] == 'Created':
    #                     if 'sensorid' in response:
    #                         self.forwarding_params = response['sensorid']
    #                         self.api_device.postprocessing['forwarding_params'] = self.forwarding_params
    #                         std_out(f'New sensor ID in {self.forwarding_request}\
    #                          is {self.forwarding_params}. Updating')

    #     if self.forwarding_params is not None:
    #         df = self.data.copy()
    #         df = df[df.columns.intersection(list(self.merge_sensor_metrics(ignore_empty=True).keys()))]
    #         df = clean(df, 'drop', how = 'all')

    #         if df.empty:
    #             std_out('Empty dataframe, ignoring', 'WARNING')
    #             return False

    #         # Create object
    #         ndev = Hclass(did = self.forwarding_params)
    #         post_ok = ndev.post_data_to_device(df, chunk_size = chunk_size,
    #             dry_run = dry_run, max_retries = 2)

    #         if post_ok:
    #             # TODO Check if we like this
    #             if self.source == 'api':
    #                 self.update_latest_postprocessing()
    #             std_out(f'Posted data for {self.params.id}', 'SUCCESS')
    #         else:
    #             std_out(f'Error posting data for {self.params.id}', 'ERROR')
    #         return post_ok

    #     else:
    #         std_out('Empty forwarding information', 'ERROR')
    #         return False

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
        if self.data is None:
            logger.error('Cannot export null data')
            return False
        if file_format == 'csv':
            return export_csv_file(path, str(self.paramsParsed.id), self.data, forced_overwrite = forced_overwrite)
        else:
            # TODO Make a list of supported formats
            return NotImplementedError (f'Not supported format. Formats: [csv]')

    async def post(self, columns = 'sensors', clean_na = 'drop', chunk_size = 500,\
        dry_run = False, max_retries = 2, with_postprocessing = False):
        '''
        Posts data via handler post method.
            Parameters
            ----------
            columns: str, optional
                ''
                'metrics' or 'sensors'. Empty '' means 'all'
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
            with_postprocessing: boolean
                False
                Update postprocessing information
        Returns
        ----------
            boolean
            True if posted ok, False otherwise
        '''

        post_ok = await self.handler.post_data(columns=columns, \
            rename = self._rename, clean_na = clean_na, chunk_size = chunk_size, \
            dry_run = dry_run, max_retries = max_retries)

        if post_ok: logger.info(f'Posted data for {self.paramsParsed.id}')
        else: logger.error(f'Error posting data for {self.paramsParsed.id}')

        # Post info if requested. It should be updated elsewhere
        if with_postprocessing and post_ok and not dry_run:
            post_ok &= self.update_postprocessing(dry_run=dry_run)

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
        if not self.postprocessing_updated:
            logger.info(f'Postprocessing is not up to date')
            return False

        post_ok = self.handler.patch_postprocessing(dry_run=dry_run)

        if post_ok: logger.info(f"Postprocessing posted for device {self.paramsParsed.id}")
        return post_ok
