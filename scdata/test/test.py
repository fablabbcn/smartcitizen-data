""" Main implementation of the class Test """

from os import makedirs
from os.path import join, exists
from shutil import copyfile, rmtree, make_archive
from traceback import print_exc
from datetime import datetime, timedelta
import json
import folium
import asyncio
from re import sub
from pydantic import TypeAdapter, BaseModel, ConfigDict, model_serializer
from typing import Optional, List, Dict, Any

from scdata.tools.custom_logger import logger
from scdata.tools.date import localise_date
from scdata.tools.find import find_by_field
from scdata.io import read_csv_file, export_csv_file
from scdata._config import config
from scdata import Device
from scdata.models import TestOptions

class Test(BaseModel):

    from .plot import (ts_plot, device_metric_map, path_plot,
        scatter_plot, ts_scatter,
        # ts_iplot, scatter_iplot, heatmap_iplot,
        heatmap_plot,
        box_plot, ts_dendrogram,
        ts_dispersion_plot, ts_dispersion_grid,
        scatter_dispersion_grid)
        #, report_plot, cat_plot, violin_plot)

    if config._ipython_avail:
        from .plot import ts_uplot, ts_dispersion_uplot
    from .export import to_csv, to_html
    from .tools import combine, prepare, history
    from .dispersion import dispersion_analysis, dispersion_summary
    from .checks import get_common_channels

    model_config = ConfigDict(arbitrary_types_allowed = True)
    name: str
    path: str = ''
    devices: List[Device] = []
    options: TestOptions = TestOptions()
    # TODO - Define test types based on enum
    # dev
    # deployment...
    type: str = 'dev'
    new: bool = False
    loaded: bool = False
    force_recreate: bool = False
    # results: List[TestResult] = []

    def model_post_init(self, __context) -> None:

        if self.__check_tname__(self.name):
            self.__set_tname__(self.name)

        if self.new or self.force_recreate:
            logger.info('New test')
            self.create()
        else:
            with open(join(self.path, 'test.json'), 'r') as file:
                tj = json.load(file)

            self.devices = TypeAdapter(List[Device]).validate_python(tj['devices'])
            self.options = TypeAdapter(TestOptions).validate_python(tj['options'])
            self.type = tj['meta']['type']
            if self.name != tj['meta']['name']:
                raise ValueError('Name not matching')

            if self.path != tj['meta']['path']:
                raise ValueError('Path not matching')

        # TODO
        # Dispersion analysis
        # self.dispersion_df = None
        # self._dispersion_summary = None
        # self.common_channels = None

        logger.info(f'Test {self.name} initialized')

    def __str__(self):
        return self.__full_name__

    def __set_tname__(self, name):
        current_date = datetime.now()
        self.name = f'{current_date.year}_{str(current_date.month).zfill(2)}_{name}'
        self.path = join(config.paths['processed'], str(current_date.year), \
                str(current_date.month).zfill(2), self.name)

        logger.info (f'Full Name: {self.name}')

    def __check_tname__(self, name):
        test_log = self.history()
        test_logn = list(test_log.keys())

        if not any([name in tlog for tlog in test_logn]):
            logger.info ('Test is new')
            self.new = True
            return name
        else:
            self.new = False
            undef_test = True
            while undef_test:
                # Wait for input
                possible_names = list()
                logger.info ('Possible tests found:')
                for ctest in test_logn:
                    if name in ctest:
                        possible_names.append(test_logn.index(ctest) + 1)
                        logger.info (str(test_logn.index(ctest) + 1) + ' --- ' + ctest)
                logger.info ('// --- \\\\')
                if len(possible_names) == 1:
                    which_test = str(possible_names[0])
                else:
                    which_test = input('Similar tests found, please select one or input other name [New]: ')

                if which_test == 'New':
                    new_name = input('Enter new name: ')
                    break
                elif which_test.isdigit():
                    if int(which_test) in possible_names:
                        self.name = test_logn[int(which_test)-1]
                        self.path = test_log[self.name]['path']
                        logger.info(f'Test full name, {self.name}')
                        return False
                    else:
                        logger.error("Type 'New' for other name, or test number in possible tests")
                else:
                    logger.error("Type 'New' for other name, or test number")

            if self.__check_tname__(new_name):
                self.__set_tname__(new_name)

    def create(self):
        # Create folder structure under data subdir
        if not exists(self.path):
            logger.info('Creating new test')
            makedirs(self.path)
        else:
            if not self.force_recreate:
                logger.error (f'Test already exists with this name. \
                    Full name: {self.name}. Maybe force_recreate = True?')
                return None
            else:
                logger.info (f'Overwriting test. Full name: {self.name}')

        # TODO Remove
        # self.__preprocess__()
        self.__dump__()

        logger.info (f'Test creation finished. Name: {self.name}')
        return self.name

    def purge(self):
        # Check if the folder structure exists
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
                         'metrics': [metric.model_dump() for metric in device.metrics],
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

        fname_t = join(self.path.strip(f'{self.full_name}')[:-1], self.full_name + f'_{selection}')
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

        if self.options.cache:
            cache_dir = join(self.path, 'cached')
            if not exists(cache_dir):
                    logger.info('Creating path for exporting cached data...')
                    makedirs(cache_dir)
                    logger.info(f'Cache will be available in: {cache_dir}')

        for device in self.devices:
            logger.info(f"Loading device {device.id}")
            # Check for cached data
            device_cache_path = ''
            if self.options.cache:
                tentative_path = join(self.path, 'cached', f'{device.id}.csv')
                if exists(tentative_path):
                    device_cache_path = tentative_path

            # Load device (no need to go async, it's fast enough)
            await device.load(cache=device_cache_path)

            if self.options.cache and device.loaded and not device.data.empty:
                if device.export(cache_dir, forced_overwrite = True, file_format = 'csv'):
                    logger.info(f'Device {device.id} cached')

        logger.info('Test load done')

        self.loaded = all([d.loaded for d in self.devices])
        return self.loaded