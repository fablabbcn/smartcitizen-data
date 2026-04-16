''' Main implementation of class Device '''

import os
from collections.abc import Iterable
from importlib import import_module
from io import StringIO
from json import dumps, loads
from os.path import basename, exists, join
from typing import Dict, List, Optional
from urllib.parse import urlparse

from numpy import nan
from pandas import DataFrame, Series, Timedelta, to_timedelta
from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic_core import ValidationError

from scdata._config import config
from scdata.io import export_csv_file, read_csv_file
from scdata.io.device_api import *
from scdata.models import (APIParams, Blueprint, CalculatedChannel, Check, CSVParams,
                           DeviceOptions, Sensor, Source)
from scdata.tools.custom_logger import logger
from scdata.tools.date import localise_date
from scdata.tools.dictmerge import dict_fmerge
from scdata.tools.find import find_by_field
from scdata.tools.lazy import LazyCallable
from scdata.tools.series import (count_nas, infer_sampling_rate, mode_ratio,
                                 normalize_central, rolling_deltas)
from scdata.tools.tree import topological_sort
from scdata.tools.units import get_units_convf
from scdata.tools.url_check import url_checker

try:
    import bokeh
    import panel
except ModuleNotFoundError:
    bokeh_available = False
    pass
else:
    bokeh_available = True

if bokeh_available:
    from scdata.plot.ts_panel import TimeSeriesPanel

try:
    import awswrangler as wr
    import boto3
    import botocore
except ModuleNotFoundError:
    boto_available = False
    pass
else:
    boto_available = True

try:
    from branca import element
    from folium import Circle
except ModuleNotFoundError:
    map_plotting_available = False
    pass
else:
    map_plotting_available = True

from timezonefinder import TimezoneFinder

tf = TimezoneFinder()

class Device(BaseModel):
    ''' Main implementation of the device class '''

    from scdata.plot import box_plot  # ts_iplot, scatter_iplot, heatmap_iplot,
    from scdata.plot import (heatmap_plot, scatter_dispersion_grid,
                             scatter_plot, ts_dendrogram, ts_dispersion_grid,
                             ts_dispersion_plot, ts_plot, ts_scatter)
        #, report_plot, cat_plot, violin_plot)
    if map_plotting_available:
        from scdata.plot import device_metric_map, path_plot

    if config._ipython_avail:
        from scdata.plot import ts_dispersion_uplot, ts_uplot

    model_config = ConfigDict(arbitrary_types_allowed = True)

    blueprint: str = None
    override_url_blueprint: bool = False
    source: Source = Source()
    options: DeviceOptions = DeviceOptions()
    params: object = None
    params_parsed: object = None
    channels: List[CalculatedChannel] = []
    checks: List[Check] = []
    meta: dict = dict()
    hclass: object = None
    handler: object = None
    data: DataFrame = DataFrame()
    qc_data: DataFrame = DataFrame()
    loaded: bool = False
    processed: bool = False
    checked: bool = False
    postprocessing_updated: bool = False
    quality_metrics: dict = dict()

    def model_post_init(self, __context) -> None:

        '''
        Creates an instance of device. Devices are objects that contain sensors data, channels
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
            A dictionary containing information about the device itself.
            Depending on the blueprint, this params needs to have different data.
            If not all the data is present, the corresponding blueprint's default will be used

        Returns
        ----------
            Device object
        '''

        # Set handler
        self.__set_handler__()

        # Set blueprint
        if self.handler.blueprint_url is not None and not self.override_url_blueprint:
            logger.info("Checking blueprint in URL")
            if url_checker(self.handler.blueprint_url):
                logger.info(f'Loading postprocessing blueprint from:\n{self.handler.blueprint_url}')
                self.blueprint = basename(urlparse(self.handler.blueprint_url).path).split('.')[0]
                self.__set_blueprint_attrs__(self.handler.properties)
        elif self.blueprint is not None:
            logger.info("Using defined blueprint")
            if self.blueprint not in config.blueprints:
                raise ValueError(f'Specified blueprint {self.blueprint} is not in available blueprints')
            self.__set_blueprint_attrs__(config.blueprints[self.blueprint])
        else:
            raise ValueError(f'Specified blueprint url {self.handler.blueprint_url} is not valid')

        logger.info(f'Device {self.params_parsed.id} initialised')

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

            self.params_parsed = TypeAdapter(APIParams).validate_python(self.params)

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

            self.params_parsed = TypeAdapter(CSVParams).validate_python(self.params)
        else:
            # TODO Add handler here
            raise NotImplementedError(f'No handler for {self.source.type} yet')

        if self.hclass is not None:
            self.handler = self.hclass(params = self.params_parsed)
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
        for sensor in self._sensors:
            if sensor.id is not None:
                _ch = find_by_field(config.names[self.source.handler], sensor.id, 'id')
                if _ch:
                    self._rename[sensor.name] = _ch.name
            else:
                logger.warning(f'Sensor {sensor.name} has no id')

        # Calculated channels stay the same
        for channel in self.channels:
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
        return self.params_parsed.id

    def add_channel(self, calculated_channel = dict()):
        '''
        Add a CalculatedChannel to the device to be processed by a callable function
        Parameters
        ----------
            calculated_channel: dict
            Empty dict
            Description of the new channel to be added. It only adds it to
            Device.channels, but does not calculate anything yet.
            The channels dict needs to follow the format:
                calculated_channel = {
                    'id': Optional[int] = None
                        <sensor_id> (in an API for instance)
                    'description': Optional[str] = ''
                        <description>
                    'name': [str]
                        <channel_name>
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
        d.calculated_channel(calculated_channel={
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
            True if calculated_channel added
        '''

        if 'channels' not in vars(self):
            logger.info(f'Device {self.params_parsed.id} has no channels yet')
            self.channels = list()

        _calculated_channel = TypeAdapter(CalculatedChannel).validate_python(calculated_channel)

        if self.__check_callable__(_calculated_channel.module, _calculated_channel.function):
            self.channels.append(_calculated_channel)

            logger.info(f"Channel {_calculated_channel.name} added to device {self.id}")
            self._rename[_calculated_channel.name] = _calculated_channel.name
            return True

    def del_channel(self, channel_name = ''):
        if 'channels' not in vars(self):
            raise ValueError('Device has no calculated channels')

        m = find_by_field(self.channels, channel_name, 'name')
        if m:
            self.channels.remove(m)
        else:
            logger.warning(f'Channel {channel_name} not in device {self.id}')
            return False
        if channel_name in self.data.columns:
            self.data.__delitem__(channel_name)

        if channel_name not in self.data and find_by_field(self.channels, channel_name, 'name') is None:
            logger.info(f'Channel {channel_name} removed from {self.id}')
            return True
        return False

    async def load(self, cache=None, convert_units=True,
    convert_names=True, ignore_error = True):
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
        ignore_error: bool
            Default: True
            Ignore if the cache does not exist
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
        limit = self.options.limit
        channels = self.options.channels
        dateformat = self.options.dateformat
        cached_data = DataFrame()

        # Only case where cache makes sense
        if self.source.type == 'api':
            if cache is not None and cache:
                if not exists(cache):
                    if not ignore_error:
                        raise FileExistsError(f'Cache does not exist: {cache}')
                    else:
                        logger.warning(f'Cache file does not exist: {cache}')
                else:
                    if cache.endswith('.csv') or cache.endswith('.csv.gz'):
                        cached_data = read_csv_file(
                            path = cache,
                            timezone = timezone,
                            frequency = frequency,
                            clean_na = clean_na,
                            resample = resample,
                            index_name = 'TIME')
                    else:
                        raise NotImplementedError(f'Cache needs to be a .csv file. Got {cache}.')

                # Make request with a logical min_date
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
                limit = limit,
                channels = channels,
                resample = resample)
        else:
            self.handler.get_data(
                min_date = min_date,
                max_date = max_date,
                frequency = frequency,
                clean_na = clean_na,
                resample = resample,
                dateformat = dateformat)

        # In principle this links both dataframes as they are unmutable
        self.data = self.handler.data

        # Wrap it all up
        self.loaded = self.__load_wrapup__(cached_data=cached_data)
        self.processed = False

        return self.loaded

    def __load_wrapup__(self, cached_data=None):

        if self.data is not None:
            if not self.data.empty:
                # Convert names
                self.__convert_names__()
                # Convert units
                self.__convert_units__()

                self.postprocessing_updated = False

            else:
                logger.info('Empty dataframe in loaded data. Waiting for cache...')

        if cached_data is not None:
            if not cached_data.empty:
                logger.info('Cache exists')
                self.data = self.data.combine_first(cached_data)

        return not self.data.empty

    def __convert_names__(self):
        if not self.options.convert_names: return
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
        if not self.options.convert_units: return

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
        # Check if the calculated_channel contains a custom module
        lazy_name = f"{module}.{function}"

        try:
            funct = LazyCallable(lazy_name)
        except ModuleNotFoundError:
            logger.error(f'Callable {lazy_name} not available')
            return False
        else:
            return True
        return False

    def process(self, only_new=False, channels_list=None):
        '''
        Processes devices calculated channels, either added by the blueprint definition
        or the addition using Device.calculated_channel(). See help(Device.calculated_channel) for
        more information.

        Parameters
        ----------
        only_new: boolean
            False
            To process or not the existing channels in the Device.data that are
            defined in Device.channels
        channels_list: list
            None
            List of channels to process. If none, processes all
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

        if 'channels' not in vars(self):
            logger.warning(f'Device {self.params_parsed.id} has nothing to process. Skipping')
            return process_ok

        logger.info(f'Processing device {self.params_parsed.id}')
        if channels_list is None:
            _channels_list = [channel.name for channel in self.channels]
        else: _channels_list = channels_list

        if not _channels_list:
            logger.warning('Nothing to process')
            return process_ok

        # Sort channels
        logger.info('Sorting channels...')
        self.channels = topological_sort(self.channels)

        for channel in self.channels:
            logger.info('---')
            if channel.name not in _channels_list: continue
            logger.info(f'Processing {channel.name}')

            if only_new and channel.name in self.data:
                logger.info(f'Skipping. Already in device')
                continue

            if self.__check_callable__(channel.module, channel.function):
                funct = LazyCallable(f"{channel.module}.{channel.function}")
            else:
                process_ok &= False
                logger.error('Problem adding lazy callable to channels list')
                continue

            args, kwargs = list(), dict()
            if 'args' in vars(channel):
                if channel.args is not None: args = channel.args
            if 'kwargs' in vars(channel):
                if channel.kwargs is not None: kwargs = channel.kwargs

            try:
                process_result = funct(self.data, *args, **kwargs)
            except KeyError as e:
                logger.error('Cannot process requested function with data provided', exc_info=e)
                process_ok = False
                pass
            else:
                # If the result is None, might be for many reasons and shouldn't collapse the process_ok
                if process_result is not None:
                    if 'ERROR' in process_result.status_code.name:
                        # We got an error during the processing
                        logger.error(process_result.status_code.name)
                        process_ok &= False
                    elif 'WARNING' in process_result.status_code.name:
                        # In this case there is no data to put into the result
                        # but there is no reason to make deny process_ok
                        logger.warning(process_result.status_code.name)
                        process_ok &= True
                    elif 'SUCCESS' in process_result.status_code.name:
                        if isinstance(process_result.data, DataFrame):
                            if len(process_result.data.columns) == 1:
                                self.data[f'{channel.name}'] = process_result.data
                            for col in process_result.data.columns:
                                if 'conc' in col:
                                    self.data[f'{channel.name}'] = process_result.data[col]
                                else:
                                    self.data[f'{channel.name}_{col}'] = process_result.data[col]
                        elif isinstance(process_result.data, Series):
                            self.data[channel.name] = process_result.data
                        else:
                            logger.error("Not supported format for data results, ignoring")
                        logger.info(process_result.status_code.name)
                        process_ok &= True

        if process_ok:
            logger.info('---')
            logger.info(f"Device {self.params_parsed.id} processed")
            self.processed = process_ok

        return self.processed

    def health_checks(self, resample=None, get_min=False, get_max=False, remake_qc_data=True):
        if not self.loaded:
            logger.error('Need to load first (device.load())')
            return False

        checked_ok = True
        if remake_qc_data:
            self.qc_data = DataFrame()

        if resample is not None:
            df = self.data.resample(resample).mean().copy()
        else:
            df = self.data.copy()

        for check in self.checks:
            logger.info('---')
            logger.info(f'Checking {check.name}')

            if self.__check_callable__(check.module, check.function):
                funct = LazyCallable(f"{check.module}.{check.function}")
            else:
                checked_ok &= False
                logger.error('Problem adding lazy callable to checks list')
                continue

            args, kwargs = list(), dict()
            if 'args' in vars(check):
                if check.args is not None:
                    args = check.args
            if 'kwargs' in vars(check):
                if check.kwargs is not None:
                    kwargs = check.kwargs

            try:
                check_result = funct(df, *args, **kwargs)
            except KeyError as e:
                logger.error('Cannot process requested function with data provided', exc_info=e)
                checked_ok = False
                pass
            else:
                # If the result is None, might be for many reasons and shouldn't collapse the process_ok
                if check_result is not None:

                    if 'ERROR' in check_result.status_code.name:
                        # We got an error during the processing
                        logger.error(check_result.status_code.name)
                        checked_ok &= False
                    elif 'WARNING' in check_result.status_code.name:
                        # In this case there is no data to put into the result
                        # but there is no reason to make deny checked_ok
                        logger.warning(check_result.status_code.name)
                        checked_ok &= True
                    elif 'SUCCESS' in check_result.status_code.name:
                        if isinstance(check_result.data, DataFrame):
                            for col in check_result.data.columns:
                                df[f'{check.name}{col}'] = check_result.data.loc[:, col]
                                # Check if we store QC
                                if check.store_qc:
                                    self.quality_metrics[f'{check.name}{col}'] = df[f'{check.name}{col}'].mean()

                                # Add clean mask
                                if check.clean:
                                    if f'CLEAN{col}' not in df.columns:
                                        df.loc[:, f'CLEAN{col}'] = False
                                    else:
                                        logger.info(f'Already clean col for {col}')
                                    df.loc[df[f'{check.name}{col}'], f'CLEAN{col}'] = True

                            # Put it back
                            self.qc_data = self.qc_data.combine_first(df)
                        else:
                            logger.error("Not supported format for data results. Ignoring")
                        logger.info(check_result.status_code.name)
                        checked_ok &= True

        # Check if we need min/max metrics
        if get_min:
            logger.info('Adding min values')
            df_min = self.data.resample(resample).min().copy()
            min_names = {col: f'MIN__{col}' for col in df_min.columns}
            df_min.rename(columns=min_names, inplace=True)
            self.qc_data = self.qc_data.combine_first(df_min)
        if get_max:
            logger.info('Adding max values')
            df_max = self.data.resample(resample).max().copy()
            max_names = {col: f'MAX__{col}' for col in df_max.columns}
            df_max.rename(columns=max_names, inplace=True)
            self.qc_data = self.qc_data.combine_first(df_max)

        # Check if we need to clean
        needs_cleaning = any([check.clean for check in self.checks])
        if needs_cleaning:
            logger.info('Cleaning qc dataset')
            cols_clean = [col for col in self.qc_data.columns if 'CLEAN__' in col]
            for col in cols_clean:
                col_name = col.replace('CLEAN__', '')
                logger.info(f'Cleaning qc data based for {col_name}')
                self.qc_data.loc[self.qc_data[col], col_name] = nan
                if get_min:
                    self.qc_data.loc[self.qc_data[col], f'MIN__{col_name}'] = nan
                if get_max:
                    self.qc_data.loc[self.qc_data[col], f'MAX__{col_name}'] = nan

        if checked_ok:
            logger.info('---')
            logger.info(f"Device {self.id} checked")
            self.checked = checked_ok

        return self.checked

    @property
    def sensors(self):
        return self._sensors

    def update_postprocessing_date(self):
        # This function updates the postprocessing date with the latest logical date
        latest_postprocessing = None
        if self.loaded:
            # If device was loaded (data not empty)
            if self.processed:
                # If device was processed, new postprocessing is the last reading rounded up with frequency
                latest_postprocessing = localise_date(self.data.index[-1] + to_timedelta(self.options.frequency), 'UTC')
                logger.info(f'Updating latest_postprocessing to {latest_postprocessing}')
            else:
                logger.info(f'Cannot update latest_postprocessing. Device was loaded but not processed')
        else:
            # If device was not loaded, increase the postprocessing limited to last_reading_at
            latest_postprocessing = min(self.handler.json.last_reading_at, self.options.max_date)
            logger.info(f'Updating latest_postprocessing to {latest_postprocessing}')

        if latest_postprocessing is None:
            return False

        if self.handler.update_latest_postprocessing(latest_postprocessing):
            # Consider the case of no postprocessing, to avoid making the whole thing false
            if latest_postprocessing.to_pydatetime() == self.handler.latest_postprocessing or self.handler.json.postprocessing is None:
                self.postprocessing_updated = True
            else:
                self.postprocessing_updated = False

        return self.postprocessing_updated

    def export(self, path, forced_overwrite = False, file_format = 'csv', gzip=False):
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
            return export_csv_file(path, str(self.params_parsed.id), self.data, forced_overwrite = forced_overwrite, gzip=gzip)
        else:
            # TODO Make a list of supported formats
            return NotImplementedError (f'Not supported format. Formats: [csv]')

    async def post(self, columns = 'sensors', clean_na = 'drop', chunk_size = 500,\
        dry_run = False, max_retries = 2, with_postprocessing = False, delay_between_posts=None):
        '''
        Posts data via handler post method.
            Parameters
            ----------
            columns: str, optional
                ''
                'channels' or 'sensors'. Empty '' means 'all'
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
            dry_run = dry_run, max_retries = max_retries, delay_between_posts=delay_between_posts)

        if post_ok: logger.info(f'Posted data for {self.params_parsed.id}')
        else: logger.error(f'Error posting data for {self.params_parsed.id}')

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

        if post_ok: logger.info(f"Postprocessing posted for device {self.params_parsed.id}")
        return post_ok

    def backup_to_storage(self, mode='append', path='devices', add_qc_data=False, add_qc_metrics=False):
        """
        Backup device data into S3 storage (requires S3_DATA_BUCKET env
         variable set).
            Parameters
            ----------
            mode: str
                'append'
                How to handle awswrangler to_parquet() storage
            path: str
                'devices'
                Path for backup directory
                "s3://{os.environ['S3_DATA_BUCKET']}/{path}/{self.id}/data/"
            add_qc_data: bool
                False
                Add quality dataframe to export as parquet
            add_qc_metrics: bool
                False
                Add quality metrics to export as json
        Returns
        ----------
            False or response from awswrangler
        """
        if self.data.empty:
            logger.error("Device data empty")
            return False

        if 'S3_DATA_BUCKET' not in os.environ:
            logger.error("S3_DATA_BUCKET not set in environment")
            return False

        # TODO Add more formats
        responses = {}
        if boto_available:
            self.data['TIME']=self.data.index
            target_path = f"s3://{os.environ['S3_DATA_BUCKET']}/{path}/{self.id}/data/"
            responses['data_upload'] = wr.s3.to_parquet(df=self.data, path=target_path, dataset=True, mode=mode)

            s3 = boto3.resource('s3')
            s3object = s3.Object(f"{os.environ['S3_DATA_BUCKET']}", f"{path}/{self.id}/metadata.json")
            s3object.put(
                Body=(bytes(self.handler.json.model_dump_json().encode('utf-8')))
            )

            if add_qc_data:
                if self.qc_data.empty:
                    logger.error("Device qc_data empty")
                else:
                    self.qc_data['TIME']=self.qc_data.index
                    target_path = f"s3://{os.environ['S3_DATA_BUCKET']}/{path}/{self.id}/qc_data/"
                    responses['qc_data_upload'] = wr.s3.to_parquet(df=self.qc_data, path=target_path, dataset=True, mode=mode)

                    # TODO Test if this works
                    if add_qc_metrics:
                        s3object = s3.Object(f"{os.environ['S3_DATA_BUCKET']}", f"{path}/{self.id}/quality_metrics.json")
                        s3object.put(
                            Body=(bytes(dumps(self.quality_metrics).encode('utf-8')))
                        )

            return responses

    def load_from_storage(self, path='devices', load_qc_data=False, load_qc_metrics=False):
        """
        Load device data from S3 storage (requires AWS env
         variable set).
            Parameters
            ----------
            path: str
                'devices'
                Path for backup directory
                "s3://{os.environ['S3_DATA_BUCKET']}/{path}/{self.id}/data/"
        Returns
        ----------
            S3 bucket url if successful, False otherwise
        """

        if 'S3_DATA_BUCKET' not in os.environ or \
            'AWS_ACCESS_KEY_ID' not in os.environ or \
            'AWS_SECRET_ACCESS_KEY' not in os.environ or \
            'AWS_REGION' not in os.environ:

            logger.error("Missing environment variables. S3_DATA_BUCKET, AWS_ACCESS_KEY_ID, \
                AWS_SECRET_ACCESS_KEY and AWS_REGION need to be set.")

            return False

        if boto_available:
            session = boto3.Session(aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'])
            s3_url = f"s3://{os.environ['S3_DATA_BUCKET']}/{path}/{self.id}/data/"
            logger.info(f"Loading data from: {s3_url}")

            self.data = wr.s3.read_parquet(s3_url, boto3_session=session, dataset=True)
            self.data.set_index('TIME', inplace=True)
            self.data.sort_index(inplace=True)
            self.data = self.data[~self.data.index.duplicated(keep='first')]

            self.loaded = True

            if load_qc_data:
                qc_data_s3_url = f"s3://{os.environ['S3_DATA_BUCKET']}/{path}/{self.id}/qc_data/"
                logger.info(f"Loading qc_data from: {qc_data_s3_url}")

                self.qc_data = wr.s3.read_parquet(qc_data_s3_url, boto3_session=session, dataset=True)
                self.qc_data.set_index('TIME', inplace=True)
                self.qc_data.sort_index(inplace=True)
                self.qc_data = self.qc_data[~self.qc_data.index.duplicated(keep='first')]

            if load_qc_metrics:
                s3 = boto3.resource('s3')
                try:
                    qc_metrics_s3_url = f"{os.environ['S3_DATA_BUCKET']}/" + f"{path}/{self.id}/quality_metrics.json"
                    logger.info(f"Loading qc_metrics from: {qc_metrics_s3_url}")

                    qc_metrics = s3.Object(f"{os.environ['S3_DATA_BUCKET']}", f"{path}/{self.id}/quality_metrics.json").get()
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "NoSuchKey":
                        # The object does not exist.
                        logger.error('No qc_metrics available')
                    else:
                        # Something else has gone wrong.
                        logger.error(e.response)
                        error = True
                else:
                    # The object does exist.
                    response = loads(qc_metrics['Body'].read().decode('utf-8'))
                    self.quality_metrics = response
            logger.info('Done')

            return s3_url
        else:
            logger.error("Boto not available. Install awswrangler")
            return False

    def get_series_dict(self, frequency=None, plot_qc_data=False):
        if plot_qc_data:
            df = self.qc_data.copy()
        else:
            df = self.data.copy()

        df.index = df.index.tz_convert('UTC').tz_localize(None)

        if frequency is not None:
            df = df.resample(frequency).mean()
        return {
            f"{self.id}:{col}": df[col]
            for col in df.columns
        }

    def ts_panel(self, frequency='10Min', channels=None, plot_qc_data=False, **kwargs):
        '''
            Returns a panel for interactive plotting
            ---
            frequency: str
                Default: 10Min
                Add a resample to the series to reduce
            width: int
                Default: 800
                Max width of each subplot (resizable to max width of window below that)
            height: int
                Default: 400
                Height of each subplot
        '''
        if bokeh_available:
            return TimeSeriesPanel(
                self.get_series_dict(frequency=frequency, plot_qc_data=plot_qc_data),
                channels=channels,
                device_id=self.id,
                **kwargs
            ).view()
        else:
            logger.error("Bokeh not available. Install with 'pip install scdata[plotting]' or 'pip install bokeh panel'")
            return False