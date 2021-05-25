from scdata.utils import std_out, localise_date
from scdata.io import read_csv_file, export_csv_file
from scdata.device import Device
from os import makedirs
from os.path import join, exists
import yaml
from datetime import timedelta

def load(self, options = dict()):

    '''
        Loads the test data and the different devices

        Parameters:
        -----------
            options: dict()

                load_cached_api: bool
                Default: config.data['data']['load_cached_api']
                Load or not cached data from the API in previous test loads

                store_cached_api: bool
                Default: config.data['data']['store_cached_api']
                Cache or not newly downloaded API data for future test loads

                clean_na: String
                Default: None
                Clean NaN as pandas format. Possibilities: 'fill_na', 'drop_na' or None

                frequency: String (timedelta format: https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags)
                Default: 1Min
                Frequency to load or request data

                min_date: String or datetime
                Default: None
                Minimum data to load data from

                max_date: String or datetime
                Default: None
                Maximum date to load data to
        Returns
        ----------
            None
    '''

    # Load descriptor
    std_out(f'Loading test {self.full_name}')
    if not exists(self.path):
        std_out('Test does not exist with that name. Have you already created it? Hint: test.create()', 'ERROR')
        return

    with open(join(self.path, 'test_description.yaml'), 'r') as descriptor_file:
        self.descriptor = yaml.load(descriptor_file, Loader = yaml.FullLoader)

    # Add devices
    for key in self.descriptor['devices'].keys():
        self.devices[key] = Device(self.descriptor['devices'][key]['blueprint'],
                                   self.descriptor['devices'][key])

    # Set options
    self.__set_options__(options)

    std_out (f'Using options: {self.options}')

    for key in self.devices.keys():

        device = self.devices[key]
        std_out('---------------------------')
        std_out(f'Loading device {device.id}')

        min_date_device = localise_date(device.min_date, device.location)
        max_date_device = localise_date(device.max_date, device.location)

        # If device comes from API, pre-check dates
        if device.source == 'api':

            if device.location is None:
                device.location = device.api_device.get_device_timezone()

            # Get last reading from API
            if 'get_device_last_reading' in dir(device.api_device):
                last_reading_api = localise_date(device.api_device.get_device_last_reading(),
                    device.location)

                if self.options['load_cached_api']:
                    std_out(f'Checking if we can load cached data')
                    if not device.load(options = self.options,
                        path = join(self.path, 'cached'), convert_units = False):

                        std_out(f'No valid cached data. Requesting device {device.id} to API', 'WARNING')
                        min_date_to_load = localise_date(device.options['min_date'],
                                                         device.location)
                        max_date_to_load = localise_date(device.options['max_date'],
                                                         device.location)
                        load_API = True

                    else:

                        std_out(f'Loaded cached files', 'SUCCESS')
                        std_out(f'Checking if new data is to be loaded')

                        # Get last reading from cached
                        last_reading_cached = localise_date(device.readings.index[-1], device.location)
                        std_out(f'Last cached date {last_reading_cached}')
                        std_out(f'Last reading in API {last_reading_api}')

                        # Check which dates to load
                        if max_date_device is not None:
                            std_out(f'Max date in test {max_date_device}')
                            # Check what where we need to load data from, if any
                            if last_reading_cached < max_date_device and last_reading_api > last_reading_cached + timedelta(hours=1):
                                load_API = True
                                combine_cache_API = True
                                min_date_to_load = last_reading_cached
                                max_date_to_load = min(max_date_device, last_reading_api)
                                std_out('Loading new data from API')
                            else:
                                load_API = False
                                std_out('No need to load new data from API')
                        else:
                            # If no test data specified, check the last reading in the API
                            if last_reading_api > (last_reading_cached + timedelta(hours=self.options['cached_data_margin'])):
                                load_API = True
                                combine_cache_API = True
                                min_date_to_load = last_reading_cached
                                max_date_to_load = last_reading_api
                                std_out('Loading new data from API')
                            else:
                                load_API = False
                                std_out('No need to load new data from API')
                else:
                    min_date_to_load = min_date_device
                    max_date_to_load = max_date_device
                    load_API = True
            else:
                if self.options['load_cached_api']:
                    std_out('Cannot load cached data without last reading available', 'WARNING')
                min_date_to_load = min_date_device
                max_date_to_load = max_date_device
                last_reading_api = None
                load_API = True

            # Load data from API if necessary
            if load_API:
                std_out('Downloading device from API')

                if last_reading_api is not None:

                    # Check which min date to load
                    if min_date_to_load is not None:
                        std_out('First reading requested: {}'.format(min_date_to_load))
                        if min_date_to_load > last_reading_api:
                            std_out('Discarding device. Min date requested is after last reading', 'WARNING')
                            continue
                    else:
                        std_out('Requesting all available data', 'WARNING')

                    # Check which max date to load
                    if max_date_to_load is not None:
                        std_out('Last reading requested: {}'.format(max_date_to_load))
                        if max_date_to_load > last_reading_api:
                            # Not possible to load what has not been stored
                            std_out('Requesting up to max available date in the API {}'.format(last_reading_api))
                            max_date_to_load = last_reading_api
                    else:
                        # Just put None and we will handle it later
                        std_out('Requesting up to max available date in the API {}'.format(last_reading_api))
                        max_date_to_load = last_reading_api

                # else:
                #     std_out('Discarding device. No max date available', 'WARNING')
                #     continue

                device_options = {
                                    'clean_na': self.options['clean_na'],
                                    'min_date': min_date_to_load,
                                    'max_date': max_date_to_load
                                 }

                if 'frequency' in self.options:
                    device_options['frequency'] = self.options['frequency']

                device.load(options = device_options)

        elif device.source == 'csv':

            device.load(options = self.options, path = self.path)

        if self.options['store_cached_api'] and device.loaded and device.source == 'api' and load_API:

            std_out(f'Caching files for {device.id}')

            cached_file_path = join(self.path, 'cached')
            if not exists(cached_file_path):
                std_out('Creating path for exporting cached data')
                makedirs(cached_file_path)

            if export_csv_file(cached_file_path, device.id, device.readings, forced_overwrite = True): std_out('Devices cached')

        if device.loaded: std_out(f'Device {device.id} has been loaded', 'SUCCESS')
        else: std_out(f'Could not load device {device.id}. Skipping', 'WARNING')

    self.__update_descriptor__()
    std_out('Test load done', 'SUCCESS')
    self.loaded = True
