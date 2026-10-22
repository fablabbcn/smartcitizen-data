""" Main implementation of the class Test """

import asyncio
import json
from datetime import datetime, timedelta
from os import getcwd, makedirs
from os.path import abspath, basename, dirname, exists, expanduser, isdir, join
from re import sub
from shutil import copyfile, make_archive, rmtree
from traceback import print_exc
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_serializer

from scdata import Device
from scdata._config import config
from scdata.io import export_csv_file, read_csv_file
from scdata.models import TestOptions
from scdata.tools.custom_logger import logger
from scdata.tools.date import localise_date
from scdata.tools.find import find_by_field
from scdata.tools.lazy import PLOT_EXTRA, plot_method
from .store import default_root, find, full_name, register

try:
    from branca import element
    from folium import Circle
except ModuleNotFoundError:
    map_plotting_available = False
    pass
else:
    map_plotting_available = True

class Test(BaseModel):
    # Plots: scdata.plot is imported on the first call (optional extra, see plot_method)
    box_plot = plot_method('box_plot')
    heatmap_plot = plot_method('heatmap_plot')
    scatter_dispersion_grid = plot_method('scatter_dispersion_grid')
    scatter_plot = plot_method('scatter_plot')
    ts_dendrogram = plot_method('ts_dendrogram')
    ts_dispersion_grid = plot_method('ts_dispersion_grid')
    ts_dispersion_plot = plot_method('ts_dispersion_plot')
    ts_plot = plot_method('ts_plot')
    ts_scatter = plot_method('ts_scatter')
    device_metric_map = plot_method('device_metric_map')
    path_plot = plot_method('path_plot')
    ts_uplot = plot_method('ts_uplot')
    ts_dispersion_uplot = plot_method('ts_dispersion_uplot')

    from .checks import get_common_channels
    from .dispersion import dispersion_analysis, dispersion_summary
    from .export import to_csv, to_html
    from .tools import combine, prepare

    model_config = ConfigDict(arbitrary_types_allowed = True)
    # <year>_<month>_<name>: the date prefix is added if the name doesn't have it
    name: str
    # Test folder, set when the test is saved or opened
    path: str = ''
    devices: List[Device] = []
    options: TestOptions = TestOptions()
    # TODO - Define test types based on enum
    # dev
    # deployment...
    type: str = 'dev'
    loaded: bool = False
    # results: List[TestResult] = []

    def model_post_init(self, __context) -> None:
        '''
        Creates a test in memory. It is stored with save(), and stored
        tests are opened with Test.open() (or Test.open_or_create())
        '''
        self.name = full_name(self.name)

        # TODO
        # Dispersion analysis
        # self.dispersion_df = None
        # self._dispersion_summary = None
        # self.common_channels = None

        logger.info(f'Test {self.name} initialized')

    def __str__(self):
        return self.name

    @classmethod
    def open(cls, name, root = None):
        '''
        Opens a stored test

        Parameters
        ----------
        name: str
            Name of the test, with or without the date prefix, or path
            to its folder (or test.json)
        root: str
            None
            Folder with tests to look into, besides config.paths['processed']
            and the registered tests

        Returns
        ----------
        Test. Raises FileNotFoundError if not found, or ValueError if several
        tests match the name
        '''
        if exists(join(name, 'test.json')):
            path = abspath(name)
        elif basename(name) == 'test.json' and exists(name):
            path = dirname(abspath(name))
        else:
            if root is not None: root = abspath(expanduser(root))
            found = find(name, root)
            if not found:
                raise FileNotFoundError(f'Test {name} not found. Stored tests are listed with sc.tests()')
            if len(found) > 1:
                raise ValueError(f'Several tests match {name}:\n' + '\n'.join(found) +
                                 '\nOpen one by its full name or path')
            path = found[0]

        with open(join(path, 'test.json'), 'r') as file:
            tj = json.load(file)

        test = cls(name = tj['meta']['name'],
                   type = tj['meta']['type'],
                   options = TypeAdapter(TestOptions).validate_python(tj['options']),
                   devices = TypeAdapter(List[Device]).validate_python(tj['devices']))
        test.path = path

        if abspath(tj['meta'].get('path', '')) != path:
            # The test folder was moved: keep the new location
            logger.warning(f"Test moved from {tj['meta'].get('path')} to {path}")
            test.__dump__()
            register(path)

        logger.info(f'Test {test.name} opened from {path}')
        return test

    @classmethod
    def open_or_create(cls, name, devices = None, options = None, type = 'dev', root = None):
        '''
        Opens the test if it is stored, otherwise creates and saves it

        Parameters
        ----------
        name: str
            Name of the test, with or without the date prefix
        devices, options, type:
            Of the new test. When the test is stored, its own are used
            (with a warning if the devices are different)
        root: str
            None
            Folder to store a new test, see save(). It is also looked into
            when opening

        Returns
        ----------
        Test
        '''
        try:
            test = cls.open(name, root = None if root == 'ask' else root)
        except FileNotFoundError:
            test = cls(name = name, devices = devices or [],
                       options = options or TestOptions(), type = type)
            return test.save(root = root)

        if devices is not None:
            stored = sorted((str(device.id), device.blueprint) for device in test.devices)
            given = sorted((str(device.id), device.blueprint) for device in devices)
            if stored != given:
                logger.warning(f'Test {test.name} is stored with other devices: {stored}. '
                               'Its stored devices are used. To change them, set '
                               'test.devices and test.save()')
        return test

    def save(self, root = None, overwrite = False):
        '''
        Stores the test (test.json) in its folder

        Parameters
        ----------
        root: str
            None
            Folder to store the test folder in. If None, the folder where
            it is already stored, or config.paths['processed']/<year>/<month>.
            'ask' asks where to store it (the scdata cache, the current
            folder, which is the notebook folder in jupyter, or another one)
        overwrite: bool
            False
            Replace a stored test with the same name. The cached data of
            the stored test is kept

        Returns
        ----------
        The test. Raises FileExistsError if a test with the same name is
        stored somewhere else (and overwrite is False)
        '''
        if root == 'ask': root = self.__ask_root__()

        if root is not None: path = join(abspath(expanduser(root)), self.name)
        elif self.path: path = abspath(self.path)
        else: path = join(default_root(self.name), self.name)

        if exists(path) and not isdir(path):
            raise FileExistsError(f'{path} is not a folder')

        # Saving the test in its own folder updates it
        current = abspath(self.path) if self.path else None
        conflicts = [found for found in find(self.name, dirname(path))
                     if not found == path == current]
        if path != current and exists(join(path, 'test.json')) and path not in conflicts:
            conflicts.append(path)

        if conflicts:
            if not overwrite:
                raise FileExistsError(f'Test {self.name} is already stored in:\n' + '\n'.join(conflicts) +
                                      '\nOpen it with sc.Test.open, or save with overwrite=True')
            if conflicts != [path]:
                logger.warning(f'Test {self.name} is also stored in: ' +
                               ', '.join(c for c in conflicts if c != path))
            logger.info(f'Overwriting test {self.name}')

        makedirs(path, exist_ok = True)
        self.path = path
        self.__dump__()
        register(path)

        logger.info(f'Test {self.name} saved in {self.path}')
        return self

    def __ask_root__(self):
        '''
        Asks where to store the test: the scdata cache, the current folder
        (the notebook folder in jupyter) or another folder.
        Returns the root folder, or None for the default one
        '''
        options = {'1': ('scdata cache', default_root(self.name)),
                   '2': ('current folder', getcwd()),
                   '3': ('other folder', None)}
        if self.path: options['1'] = ('where it is stored', dirname(self.path))
        print (f'Where should test {self.name} be stored?')
        for key, (description, path) in options.items():
            print (f'  {key} - {description}' + (f': {path}' if path else ''))

        while True:
            try:
                which = input('Select an option [1]: ').strip() or '1'
                if which == '3':
                    root = input('Folder: ').strip()
                    if not root: continue
                    root = abspath(expanduser(root))
                    if exists(root) and not isdir(root):
                        logger.error(f'{root} is not a folder')
                        continue
                    return root
            except (EOFError, NotImplementedError):
                # No input available (i.e. a notebook run with nbconvert)
                logger.warning(f'Input not available. Using {options["1"][1]}')
                return None

            if which == '1': return None
            if which == '2': return options['2'][1]
            logger.error('Type 1, 2 or 3')

    def __saved__(self):
        if not self.path:
            logger.error(f'Test {self.name} is not saved: use test.save()')
        return bool(self.path)

    def purge(self):
        # Check if the folder structure exists
        if not self.__saved__(): return False
        if not exists(self.path):
            logger.error('Test folder doesnt exist')
        else:
            logger.info (f'Purging cached directory in: {self.path}')
            try:
                rmtree(join(self.path, 'cached'))
            except:
                logger.error('Error while purging directory')
                pass
            else:
                logger.info (f'Purged cached folder')
                return True
        return False

    def get_device(self, device_id):
        did = find_by_field(self.devices, device_id, 'id')
        if did is None:
            logger.error(f'Device {device_id} is not in test')
        return did

    # TODO - Do we want this with asyncio?
    def process(self, only_new = False):
        '''
        Calculates all the metrics in each of the devices
        Returns True if done OK
        '''
        process_ok = True
        for device in self.devices:
            process_ok &= device.process(only_new = only_new)

        # Cosmetic output
        if process_ok: logger.info(f'Test {self.name} processed')
        else: logger.error(f'Test {self.name} not processed')

        return process_ok

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:

        return {
            'meta': {
                'name': self.name,
                'path': self.path,
                'type': self.type
            },
            'options': self.options.model_dump(),
            'devices': [{'params': device.params.model_dump(),
                         'channels': [channel.model_dump() for channel in device.channels],
                         'checks': [check.model_dump() for check in device.checks],
                         'source': device.source.model_dump(),
                         'blueprint': device.blueprint}
                         for device in self.devices]
        }

    def __dump__(self):
        with open(join(self.path, 'test.json'), 'w') as file:
            json.dump(self.ser_model(), file, indent=4)

    def compress(self, cformat = 'zip', selection = 'full'):
        '''
        Compress the test folder (or selected folder) into a defined
        format in the test.path directory

        Parameters
        ----------
        cformat
            'zip'
            String. Valid shutil.make_archive input: 'zip', 'tar',
            'gztar', 'bztar', 'xztar'
        selection
            'full'
            String. Selection of folders to compress. Either 'full',
            'cached' or 'raw'. If 'full', compresses the whole test

        Returns
        ----------
        True if all good, False otherwise
        '''
        if not self.__saved__(): return False

        if cformat not in ['zip', 'tar', 'gztar', 'bztar', 'xztar']:
            logger.error('Invalid format')
            return False

        if selection not in ['full', 'cached', 'raw']:
            logger.error('Invalid selection (valid options: full, cached, raw')
            return False

        if selection == 'full':
            _root_dir = self.path
        elif selection == 'cached':
            _root_dir = join(self.path, 'cached')
        elif selection == 'raw':
            _root_dir = join(self.path, 'raw')

        fname_t = join(dirname(abspath(self.path)), self.name + f'_{selection}')
        make_archive(fname_t, cformat, root_dir=_root_dir)

        fname = fname_t + '.' + cformat
        if not exists(fname): return False

        return fname

    async def load(self):
        '''
        Loads the test data and the different devices.

        Returns
        ----------
            None
        '''
        logger.info('Loading test...')

        cache = self.options.cache
        if cache and not self.path:
            logger.warning(f'Test {self.name} is not saved: its data is not cached (use test.save())')
            cache = False

        if cache:
            cache_dir = join(self.path, 'cached')
            if not exists(cache_dir):
                    logger.info('Creating path for exporting cached data...')
                    makedirs(cache_dir)
                    logger.info(f'Cache will be available in: {cache_dir}')

        for device in self.devices:
            logger.info(f"Loading device {device.id}")
            # Check for cached data
            device_cache_path = ''
            if cache:
                tentative_path = join(self.path, 'cached', f'{device.id}.csv')
                if exists(tentative_path):
                    device_cache_path = tentative_path

            # Load device (no need to go async, it's fast enough)
            await device.load(cache=device_cache_path)

            if cache and device.loaded and not device.data.empty:
                if device.export(cache_dir, forced_overwrite = True, file_format = 'csv', use_exports = False):
                    logger.info(f'Device {device.id} cached')

        logger.info('Test load done')

        self.loaded = all([d.loaded for d in self.devices])
        return self.loaded

    def get_series_dict(self, frequency=None):
        series = {}

        for device in self.devices:
            if device.loaded:
                series.update(device.get_series_dict(frequency))

        return series

    def ts_panel(self, frequency='10Min', **kwargs):
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
        # Plotting is an optional extra: imported here, not when scdata is imported
        try:
            from scdata.plot.ts_panel import TimeSeriesPanel
        except ImportError as error:
            raise ImportError(f'ts_panel needs the plotting libraries: {PLOT_EXTRA}') from error
        return TimeSeriesPanel(
            self.get_series_dict(frequency=frequency),
            **kwargs
        ).view()