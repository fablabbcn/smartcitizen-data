""" Main implementation of the class Test """

from os import makedirs
from os.path import join, exists
from shutil import copyfile
from traceback import print_exc
from datetime import datetime
import yaml
import json
import folium

from scdata.utils import std_out, get_tests_log
from scdata.io import read_csv_file
from scdata._config import config
from scdata.device import Device

class Test(object):

    from .plot import (ts_plot, ts_iplot, device_metric_map,
                        scatter_plot, scatter_iplot, ts_scatter,
                        heatmap_plot, heatmap_iplot,
                        box_plot, ts_dendrogram)
                        #, report_plot, cat_plot, violin_plot)
    from .export import to_csv, desc_to_html
    from .load import load
    from .utils import combine, prepare

    def __init__(self, name):
        
        self.options = {  
                        'cached_data_margin': config.data['cached_data_margin'],
                        'load_cached_api': config.data['load_cached_api'],
                        'store_cached_api': config.data['store_cached_api']
                        }

        if self.__check_tname__(name): self.__set_tname__(name)

        self.details = dict()
        self.devices = dict()

        self.descriptor = dict()
        self.models = dict()
        self.descriptor['id'] = self.full_name
        self.cached_info = dict()
        self.ready_to_model = False

        self._default_fields = {
                                'id': '',
                                'comment': '',
                                'notes': '',
                                'project': '',
                                'author': '',
                                'commit': '',
                                'devices': dict(),
                                'report': '',
                                'type_test': ''
                                }

    def __str__(self):
        return self.full_name

    def __set_tname__(self, name):
        current_date = datetime.now()
        self.full_name = f'{current_date.year}_{str(current_date.month).zfill(2)}_{name}'
        self.path = join(config.paths['processed'], str(current_date.year), \
                str(current_date.month).zfill(2), self.full_name)
        std_out (f'Full Name: {self.full_name}') 

    def __check_tname__(self, name):
        test_log = get_tests_log()
        test_logn = list(test_log.keys())

        if not any([name in tlog for tlog in test_logn]):
            return name
        else:
            undef_test = True

            while undef_test:
                
                # Wait for input
                poss_names = list()
                for ctest in test_logn: 
                    if name in ctest:
                        poss_names.append(test_logn.index(ctest) + 1)
                        std_out (str(test_logn.index(ctest) + 1) + ' --- ' + ctest, force = True)                
                which_test = input('Similar tests found, please select one or input other name [New]: ')
                
                if which_test == 'New':
                    new_name = input('Enter new name: ')
                    break
                elif which_test.isdigit():
                    if int(which_test) in poss_names: 
                        self.full_name = test_logn[int(which_test)-1]
                        self.path = test_log[self.full_name]['path']
                        std_out(f'Test full name, {self.full_name}', force = True)
                        return False
                    else:
                        std_out("Type 'New' for other name, or test number in possible tests", 'ERROR')
                else: 
                    std_out("Type 'New' for other name, or test number", 'ERROR')
            if self.__check_tname__(new_name): self.__set_tname__(new_name)

    def create(self, force = False):
        # Create folder structure under data subdir
        if not exists(self.path): 
            std_out('Creating new test')
            makedirs(self.path)
        else: 
            if not force: 
                std_out (f'Test already exists with this name. Full name: {self.full_name}. Maybe force = True?', 'ERROR')
                return None
            else: 
                std_out (f'Overwriting test. Full name: {self.full_name}')
        
        self.__update_descriptor__()
        self.__preprocess__()

        std_out (f'Test creation finished. Name: {self.full_name}', 'SUCCESS')
        return self.full_name

    def add_details(self, details):
        '''
            details: a dict containing the information about the test. Minimum of:
                - project
                - commit
                - author
                - test_type
                - report
                - comment
        '''

        for detail in details.keys(): self.details[detail] = details[detail]

    def add_devices_list(self, devices_list, blueprint):
        '''
            Convenience method to add devices from a list of api devices with a certain blueprint
            Params:
                devices_list: list
                Contains devices ids (str or int)

                blueprint: String
                Blueprint name
        '''
        if blueprint is None: return False

        if blueprint not in config.blueprints.keys():
            std_out(f'Blueprint {blueprint} not in blueprints', 'ERROR')
            return False      

        for device in devices_list:
            self.add_device(Device(blueprint = blueprint , descriptor = {'source': 'api', 
                                                                            'id': str(device)
                                                                            }
                                    )
            ) 

    def add_device(self, device):
        '''
            Adds a device to the test. The device has to be an instance of 'scdata.device.Device'
        '''
        if device.id not in self.devices.keys(): self.devices[device.id] = device
        else: std_out(f'Device {device.id} is duplicated', 'WARNING')

    def process(self, only_new = False):
        ''' 
        Calculates all the metrics in each of the devices
        Returns True if done OK
        '''
        process_ok = True
        for device in self.devices: process_ok &= self.devices[device].process(only_new = only_new)
        
        # Cosmetic output
        if process_ok: std_out(f'Test {self.full_name} processed', 'SUCCESS')
        else: std_out(f'Test {self.full_name} not processed', 'ERROR')
        
        return process_ok

    def reprocess(self):
        ''' 
        Calculates only the new metrics in each of the devices
        Returns True if done OK
        '''
        return self.process(only_new = True)     

    def __set_options__(self, options):
        if 'load_cached_api' in options.keys(): self.options['load_cached_api'] = options['load_cached_api']
        if 'store_cached_api' in options.keys(): self.options['store_cached_api'] = options['store_cached_api']
        if 'clean_na' in options.keys(): self.options['clean_na'] = options['clean_na']
        if 'frequency' in options.keys(): self.options['frequency'] = options['frequency']
        if 'min_date' in options.keys(): self.options['min_date'] = options['min_date'] 
        if 'max_date' in options.keys(): self.options['max_date'] = options['max_date']

    def __preprocess__(self):
        '''
            Processes the files for one test, given that the devices and details have been added
        '''

        std_out('Processing files')
        
        def get_raw_files():
                list_raw_files = []
                for device in self.devices.keys():
                    if self.devices[device].source == 'csv':
                        list_raw_files.append(self.devices[device].raw_data_file)
                
                return list_raw_files    
        
        def copy_raw_files(_raw_src_path, _raw_dst_path, _list_raw_files):

                try: 

                    for item in _list_raw_files:
                        s = join(_raw_src_path, item)
                        d = join(_raw_dst_path, item)
                        copyfile(s, d)

                    std_out('Copy raw files: OK', 'SUCCESS')
                    
                    return True
                
                except:
                    std_out('Problem copying raw files', 'ERROR')
                    print_exc()
                    return False
                
        def date_parser(s, a):
            return parser.parse(s).replace(microsecond=int(a[-3:])*1000)

        # Define paths
        raw_src_path = join(config.paths['data'], 'raw')
        raw_dst_path = join(self.path, 'raw')

        # Create path
        if not exists(raw_dst_path): makedirs(raw_dst_path)
        
        # Get raw files
        list_raw_files = get_raw_files()

        # Copy raw files and process data
        if len(list_raw_files):
            if copy_raw_files(raw_src_path, raw_dst_path, list_raw_files):  
            
                # Process devices
                for device_name in self.devices.keys():

                    device = self.devices[device_name]

                    if device.source == 'csv':

                        std_out ('Processing csv from device {}'.format(device.id))
                        src_path = join(raw_src_path, device.raw_data_file)
                        dst_path = join(self.path, device.processed_data_file)

                        # Load csv file, only localising and removing 
                        df = read_csv_file(src_path, device.location, device.frequency, clean_na = None, 
                                           index_name = device.sources[device.source]['index'], 
                                           skiprows = device.sources[device.source]['header_skip'])
                        df.to_csv(dst_path, sep=",")
                    
            std_out('Files preprocessed')
        std_out(f'Test {self.full_name} path: {self.path}')

    @property
    def default_fields(self):
        return self._default_fields
    
    def __update_descriptor__(self):
        if self.descriptor == {}: self.std_out('No descriptor file to update')

        for field in self._default_fields:
            if field not in self.descriptor.keys(): self.descriptor[field] = self._default_fields[field]
        
        # Add details to descriptor, or update them if there is anything in details
        for detail in self.details.keys(): self.descriptor[detail] = self.details[detail]

        # Add devices to descriptor
        for device_name in self.devices.keys():

            device = self.devices[device_name]
            
            if device.source == 'csv': 
                device.processed_data_file = self.full_name + '_' + str(device.id) + '.csv'

            dvars = vars(device).copy()
            for discvar in ['readings', 'api_device', 'options', 'loaded', 'hw_id', 'latest_postprocessing']:
                if discvar in dvars: dvars.pop(discvar)

            self.descriptor['devices'][device.id] = dvars

        # Create yaml with test description
        with open(join(self.path, 'test_description.yaml'), 'w') as yaml_file: 
            yaml.dump(self.descriptor, yaml_file)
            
        std_out ('Descriptor file updated')