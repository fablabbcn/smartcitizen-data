from pandas import (DataFrame, to_datetime, to_numeric, to_timedelta,
                    to_numeric, read_csv, DateOffset, MultiIndex)

from math import isnan
from traceback import print_exc
from requests import get, post, patch
from re import search
from io import StringIO

from geopy.distance import distance
from scdata._config import config
from scdata.utils import std_out, localise_date, clean, get_elevation, url_checker
from tzwhere import tzwhere
from datetime import date, datetime
from os import environ, urandom
from json import dumps

import binascii
from time import sleep

import sys
from tqdm import trange

tz_where = tzwhere.tzwhere(forceTZ=True)

'''
About the classes in this file:
Each of the object interacts with a separate API.
There should be at minimum the following properties:
- id: identifier against the API
- timezone: timezone
- sensors: dictionary used to convert the names to saf standard (see saf.py and blueprints.yml) {api_name: saf_name}
- data: pandas dataframe containing the data. Columns = pollutants or sensors; index = localised timestamp
Methods
- get_device_data(min_date, max_date, frequency, clean_na): returns clean pandas dataframe (self.data) with start and end date filtering, and rollup
- get_device_timezone: returns timezone for timestamp geolocalisation

The units should not be converted here, as they will be later on converted in device.py
If you want to support caching, see get_device_data in ScApiDevice
'''

class ScApiDevice:

    API_BASE_URL='https://api.smartcitizen.me/v0/devices/'

    def __init__ (self, did):

        self.id = did # the number after https://smartcitizen.me/kits/######
        self.kit_id = None # the number that defines the type of blueprint
        self.mac = None
        self.last_reading_at = None
        self.added_at = None
        self.timezone = None
        self.lat = None
        self.long = None
        self.alt = None
        self.data = None
        self.sensors = None
        self.devicejson = None
        self.postprocessing = None
        self._url = f'https://smartcitizen.me/kits/{self.id}'
        self._api_url = f'{self.API_BASE_URL}{self.id}'

    @property
    def url(self):
        return self._url

    @property
    def api_url(self):
        return self._api_url

    @staticmethod
    def new_device(name, kit_id = 26, latitude = 41.396867,  longitude = 2.194351, exposure = 'indoor', user_tags = 'Lab, Research, Experimental'):
        '''
            Creates a new device in the Smart Citizen Platform provided a name
            Parameters
            ----------
                name: string
                    Device name
                kit_id: int, optional
                    26 (SCK 2.1)
                    Kit ID - related to blueprint
                latitude: int, optional
                    41.396867
                    Latitude
                longitude: int, optional
                    2.194351
                    Longitude
                exposure: string, optional
                    'indoor'
                    Type of exposure ('indoor', 'outdoor')
                user_tags: string
                    'Lab, Research, Experimental'
                    User tags, comma sepparated
            Returns
            -------
                platform id
        '''

        if 'SC_ADMIN_BEARER' not in environ:
            std_out('Cannot post without Auth Bearer', 'ERROR')
            return

        headers = {'Authorization':'Bearer ' + environ['SC_ADMIN_BEARER'], 'Content-type': 'application/json'}

        device = {}
        try:
            device['name'] = name
        except:
            std_out('Your device needs a name!', 'ERROR')
            # TODO ask for a name
            sys.exit()

        device['device_token'] = binascii.b2a_hex(urandom(3)).decode('utf-8')
        device['description'] = ''
        device['kit_id'] = kit_id
        device['latitude'] = latitude
        device['longitude'] = longitude
        device['exposure'] = exposure
        device['user_tags'] = user_tags

        device_json = dumps(device)
        backed_device = post('https://api.smartcitizen.me/v0/devices', data=device_json, headers=headers)

        if backed_device.status_code == 200 or backed_device.status_code == 201:

            platform_id = str(backed_device.json()['id'])
            platform_url = "https://smartcitizen.me/kits/" + platform_id
            std_out(f'Device created with: \n{platform_url}', 'SUCCESS')
            return platform_id

        std_out(f'Error while creating new device, platform returned {backed_device.status_code}', 'ERROR')
        return False

    @staticmethod
    def global_search(value = None, full = False):
        """
        Gets devices from Smart Citizen API based on basic search query values,
        searching both Users and Devices at the same time.
        Global search documentation: https://developer.smartcitizen.me/#global-search
        Parameters
        ----------
            value: string
                None
                Query to fit
                For null, not_null values, use 'null' or 'not_null'
            full: bool
                False
                Returns a list with if False, or the whole dataframe if True
        Returns
        -------
            A list of kit IDs that comply with the requirements, or the full df, depending on full.
        """

        API_BASE_URL = "https://api.smartcitizen.me/v0/search?q="

        # Value check
        if value is None: std_out(f'Value needs a value, {value} supplied', 'ERROR'); return None

        query = API_BASE_URL  + f'{value}'

        try:
            qresp = get(query)

            # If status code OK, retrieve data
            if qresp.status_code == 200 or qresp.status_code == 201:
                df = DataFrame(qresp.json()).set_index('id')
            else:
                std_out('API reported {}'.format(qresp.status_code), 'ERROR')
        except:
            std_out('Failed request. Probably no connection', 'ERROR')
            pass

        if full: return df
        else: return list(df.index)

    @staticmethod
    def search_by_query(key = '', value = None, full = False):
        """
        Gets devices from Smart Citizen API based on ransack parameters
        Basic query documentation: https://developer.smartcitizen.me/#basic-searching
        Parameters
        ----------
            key: string
                ''
                Query key according to the basic query documentation. Some (not all) parameters are:
                ['id', 'owner_id', 'name', 'description', 'mac_address', 'created_at',
                'updated_at', 'kit_id', 'geohash', 'last_recorded_at', 'uuid', 'state',
                'postprocessing_id', 'hardware_info']
            value: string
                None
                Query to fit
                For null, not_null values, use 'null' or 'not_null'
            full: bool
                False
                Returns a list with if False, or the whole dataframe if True
        Returns
        -------
            A list of kit IDs that comply with the requirements, or the full df, depending on full.
        """

        API_BASE_URL = "https://api.smartcitizen.me/v0/devices/"

        # Value check
        if value is None: std_out(f'Value needs a value, {value} supplied', 'ERROR'); return None

        if value == 'null' or value == 'not_null':
             query = API_BASE_URL  + f'?q[{key}_{value}]=1'
        else:
             query = API_BASE_URL  + f'?q[{key}]={value}'

        try:
            qresp = get(query)

            # If status code OK, retrieve data
            if qresp.status_code == 200 or qresp.status_code == 201:
                df = DataFrame(qresp.json()).set_index('id')
            else:
                std_out('API reported {}'.format(qresp.status_code), 'ERROR')
        except:
            std_out('Failed request. Probably no connection', 'ERROR')
            pass

        if full: return df
        else: return list(df.index)

    @staticmethod
    def get_world_map(min_date = None, max_date = None, city = None, within = None, tags = None, tag_method = 'any', full = False):
        """
        Gets devices from Smart Citizen API with certain requirements
        Parameters
        ----------
            min_date: string, datetime-like object, optional
                None
                Minimum date to filter out the devices. Device started posted before min_date
            max_date: string, datetime-like object, optional
                None
                Maximum date to filter out the devices. Device posted after max_date
            city: string, optional
                Empty string
                City
            within: tuple
                Empty tuple
                Gets the devices within a circle center on lat, long with a radius_meters
                within = tuple(lat, long, radius_meters)
            tags: list of strings
                None
                Tags for the device (system or user). Default system wide are: indoor, outdoor, online, and offline
            tag_method: string
                'any'
                'any' or 'all'. Checks if 'all' the tags are to be included in the tags or it could be any
            full: bool
                False
                Returns a list with if False, or the whole dataframe if True
        Returns
        -------
            A list of kit IDs that comply with the requirements, or the full df, depending on full.
            If no requirements are set, returns all of them
        """

        def is_within_circle(x, within):
            if isnan(x['latitude']): return False
            if isnan(x['longitude']): return False

            return distance((within[0], within[1]), (x['latitude'], x['longitude'])).m<within[2]

        world_map = get('https://api.smartcitizen.me/v0/devices/world_map')

        df = DataFrame(world_map.json()).set_index('id')

        # Filter out dates
        if min_date is not None: df=df[(min_date > df['added_at'])]
        if max_date is not None: df=df[(max_date < df['last_reading_at'])]

        # Location
        if city is not None: df=df[(df['city']==city)]
        if within is not None:

            df['within'] = df.apply(lambda x: is_within_circle(x, within), axis=1)
            df=df[(df['within']==True)]

        # Tags
        if tags is not None:
            if tag_method == 'any':
                df['has_tags'] = df.apply(lambda x: any(tag in x['system_tags']+x['user_tags'] for tag in tags), axis=1)
            elif tag_method == 'all':
                df['has_tags'] = df.apply(lambda x: all(tag in x['system_tags']+x['user_tags'] for tag in tags), axis=1)
            df=df[(df['has_tags']==True)]

        if full: return df
        else: return list(df.index)

    def get_mac(self, update = False):
        if self.mac is None or update:
            std_out(f'Requesting MAC from API for device {self.id}')
            # Get device
            try:
                deviceR = get(self.API_BASE_URL + '{}/'.format(self.id))

                # If status code OK, retrieve data
                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    if 'hardware_info' in deviceR.json().keys(): self.mac = deviceR.json()['hardware_info']['mac']
                    std_out ('Device {} is has this MAC {}'.format(self.id, self.mac))
                else:
                    std_out('API reported {}'.format(deviceR.status_code), 'ERROR')
            except:
                std_out('Failed request. Probably no connection', 'ERROR')
                pass

        return self.mac

    def get_device_json(self, update = False):
        if self.devicejson is None or update:
            try:
                deviceR = get(self.API_BASE_URL + '{}/'.format(self.id))
                if deviceR.status_code == 429:
                    std_out('API reported {}. Retrying once'.format(deviceR.status_code),
                            'WARNING')
                    sleep(30)
                    deviceR = get(self.API_BASE_URL + '{}/'.format(self.id))

                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    self.devicejson = deviceR.json()
                else:
                    std_out('API reported {}'.format(deviceR.status_code), 'ERROR')
            except:
                std_out('Failed request. Probably no connection', 'ERROR')
                pass
        return self.devicejson

    def get_kit_ID(self, update = False):

        if self.kit_id is None or update:
            if self.get_device_json(update) is not None:
                self.kit_id = self.devicejson['kit']['id']

        return self.kit_id

    def post_kit_ID(self):
        '''
            Posts kit id to platform
        '''

        if 'SC_ADMIN_BEARER' not in environ:
            std_out('Cannot post without Auth Admin Bearer', 'ERROR')
            return

        headers = {'Authorization':'Bearer ' + environ['SC_ADMIN_BEARER'], 'Content-type': 'application/json'}

        if self.kit_id is not None:

            payload = {'kit_id': self.kit_id}

            payload_json = dumps(payload)
            response = patch(f'{self.API_BASE_URL}{self.id}',
                        data = payload_json, headers = headers)

            if response.status_code == 200 or response.status_code == 201:
                std_out(f'Kit ID for device {self.id} was updated to {self.kit_id}', 'SUCCESS')
                return True

        std_out(f'Problem while updating kit ID for device {self.id}')

        return False

    def get_device_last_reading(self, update = False):

        if self.last_reading_at is None or update:
            if self.get_device_json(update) is not None:
                self.last_reading_at = self.devicejson['last_reading_at']

        std_out ('Device {} has last reading at {}'.format(self.id, self.last_reading_at))

        return self.last_reading_at

    def get_device_postprocessing(self, update = False):

        if self.postprocessing is None or update:
            if self.get_device_json(update) is not None:
                self.postprocessing = self.devicejson['postprocessing']

                if self.postprocessing is not None:
                    # Check the url in hardware
                    if 'hardware_url' in self.postprocessing:
                        urls = url_checker(self.postprocessing['hardware_url'])
                        # If URL is empty, try prepending base url from config
                        if not urls:
                            tentative_url = f"{config._base_postprocessing_url}hardware/{self.postprocessing['hardware_url']}.{config._default_file_type}"
                        else:
                            if len(urls)>1: std_out('URLs for postprocessing recipe are more than one, trying first', 'WARNING')
                            tentative_url = urls[0]

                        self.postprocessing['hardware_url'] = tentative_url

                    std_out ('Device {} has postprocessing information:\n{}'.format(self.id, self.postprocessing))
                else:
                    std_out (f'Device {self.id} has no postprocessing information')

        return self.postprocessing

    def get_device_timezone(self, update = False):

        if self.timezone is None or update:
            latitude, longitude = self.get_device_lat_long(update)
            # Localize it

            if latitude is not None and longitude is not None:
                self.timezone = tz_where.tzNameAt(latitude, longitude, forceTZ=True)

        std_out ('Device {} timezone is {}'.format(self.id, self.timezone))

        return self.timezone

    def get_device_lat_long(self, update = False):

        if self.lat is None or self.long is None or update:
            if self.get_device_json(update) is not None:
                latidude = longitude = None
                if 'location' in self.devicejson.keys():
                    latitude, longitude = self.devicejson['location']['latitude'], self.devicejson['location']['longitude']
                elif 'data' in self.devicejson.keys():
                    if 'location' in self.devicejson['data'].keys():
                        latitude, longitude = self.devicejson['data']['location']['latitude'], self.devicejson['data']['location']['longitude']

                self.lat = latitude
                self.long = longitude

        std_out ('Device {} is located at {}, {}'.format(self.id, self.lat, self.long))

        return (self.lat, self.long)

    def get_device_alt(self, update = False):

        if self.lat is None or self.long is None:
            self.get_device_lat_long(update)


        if self.alt is None or update:
            self.alt = get_elevation(_lat = self.lat, _long = self.long)

        std_out ('Device {} altitude is {}m'.format(self.id, self.alt))

        return self.alt

    def get_device_added_at(self, update = False):

        if self.added_at is None or update:
            if self.get_device_json(update) is not None:
                self.added_at = self.devicejson['added_at']

        std_out ('Device {} was added at {}'.format(self.id, self.added_at))

        return self.added_at

    def get_device_sensors(self, update = False):

        if self.sensors is None or update:
            if self.get_device_json(update) is not None:
                # Get available sensors in platform
                sensors = self.devicejson['data']['sensors']

                # Put the ids and the names in lists
                self.sensors = dict()
                for sensor in sensors:
                    for key in config.sc_sensor_names:
                        if str(config.sc_sensor_names[key]['id']) == str(sensor['id']):
                            # IDs are not unique
                            if key in config._sc_ignore_keys: continue
                            self.sensors[sensor['id']] = key

        return self.sensors

    def convert_rollup(self, frequency):
        # Convert frequency from pandas to API's
        for index, letter in enumerate(frequency):
            try:
                aux = int(letter)
            except:
                index_first = index
                letter_first = letter
                rollup_value = frequency[:index_first]
                frequency_unit = frequency[index_first:]
                break

        for item in config._freq_conv_lut:
            if item[1] == frequency_unit:
                rollup_unit = item[0]
                break

        rollup = rollup_value + rollup_unit
        return rollup

    def get_device_data(self, min_date = None, max_date = None, frequency = '1Min', clean_na = None):

        if 'SC_ADMIN_BEARER' in environ:
            std_out('Admin Bearer found, using it', 'SUCCESS')

            headers = {'Authorization':'Bearer ' + environ['SC_ADMIN_BEARER']}
        else:
            headers = None
            std_out('Admin Bearer not found', 'WARNING')

        std_out(f'Requesting data from SC API')
        std_out(f'Device ID: {self.id}')

        rollup = self.convert_rollup(frequency)
        std_out(f'Using rollup: {rollup}')

        # Make sure we have the everything we need beforehand
        self.get_device_sensors()
        self.get_device_timezone()
        self.get_device_last_reading()
        self.get_device_added_at()
        self.get_kit_ID()

        if self.timezone is None:
            std_out('Device does not have timezone set, skipping', 'WARNING')
            return None

        # Check start date and end date
        # Converting to UTC by passing None
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.tz_convert.html
        if min_date is not None:
            min_date = localise_date(to_datetime(min_date), 'UTC').strftime('%Y-%m-%dT%H:%M:%S')
            std_out (f'Min Date: {min_date}')
        else:
            min_date = localise_date(to_datetime('2001-01-01'), 'UTC').strftime('%Y-%m-%dT%H:%M:%S')
            std_out(f"No min_date specified")

        if max_date is not None:
            max_date = localise_date(to_datetime(max_date), 'UTC').strftime('%Y-%m-%dT%H:%M:%S')
            std_out (f'Max Date: {max_date}')

        # Print stuff
        std_out('Kit ID: {}'.format(self.kit_id))
        std_out(f'Device timezone: {self.timezone}')
        if not self.sensors.keys():
            std_out(f'Device is empty')
            return None
        else: std_out(f'Sensor IDs: {list(self.sensors.keys())}')

        df = DataFrame()

        # Get devices in the sensor first
        for sensor_id in self.sensors.keys():

            # Request sensor per ID
            request = self.API_BASE_URL + '{}/readings?'.format(self.id)

            if min_date is not None: request += f'from={min_date}'
            if max_date is not None: request += f'&to={max_date}'

            request += f'&rollup={rollup}'
            request += f'&sensor_id={sensor_id}'
            request += '&function=avg'

            # Make request
            sensor_req = get(request, headers = headers)

            # Retry once in case of 429 after 30s
            if sensor_req.status_code == 429:
                std_out('Too many requests, waiting for 1 more retry', 'WARNING')
                sleep (30)
                sensor_req = get(request, headers = headers)

            flag_error = False
            try:
                sensorjson = sensor_req.json()
            except:
                std_out(f'Problem with json data from API, {sensor_req.status_code}', 'ERROR')
                flag_error = True
                pass
                continue

            if 'readings' not in sensorjson.keys():
                std_out(f'No readings key in request for sensor: {sensor_id}', 'ERROR')
                flag_error = True
                continue

            elif sensorjson['readings'] == []:
                std_out(f'No data in request for sensor: {sensor_id}', 'WARNING')
                flag_error = True
                continue

            if flag_error: continue

            try:
                dfsensor = DataFrame(sensorjson['readings']).set_index(0)
                dfsensor.columns = [self.sensors[sensor_id]]
                # dfsensor.index = to_datetime(dfsensor.index).tz_localize('UTC').tz_convert(self.timezone)
                dfsensor.index = localise_date(dfsensor.index, self.timezone)
                dfsensor.sort_index(inplace=True)
                dfsensor = dfsensor[~dfsensor.index.duplicated(keep='first')]

                # Drop unnecessary columns
                dfsensor.drop([i for i in dfsensor.columns if 'Unnamed' in i], axis=1, inplace=True)
                # Check for weird things in the data
                dfsensor = dfsensor.astype(float, errors='ignore')
                # dfsensor = dfsensor.apply(to_numeric, errors='coerce')
                # Resample
                dfsensor = dfsensor.resample(frequency).mean()
                df = df.combine_first(dfsensor)
            except:
                print_exc()
                std_out('Problem with sensor data from API', 'ERROR')
                flag_error = True
                pass
                continue

            try:
                df = df.reindex(df.index.rename('TIME'))
                df = clean(df, clean_na, how = 'all')
                self.data = df

            except:
                std_out('Problem closing up the API dataframe', 'ERROR')
                pass
                return None

        if flag_error == False: std_out(f'Device {self.id} loaded successfully from API', 'SUCCESS')
        return self.data

    def post_device_data(self, clean_na = 'drop', chunk_size = 500):
        '''
            POST self.data in the SmartCitizen API
            Parameters
            ----------
                clean_na: string, optional
                    'drop'
                    'drop', 'fill'
                chunk_size: integer
                    chunk size to split resulting pandas DataFrame for posting readings
            Returns
            -------
                True if the data was posted succesfully
        '''
        if self.data is None:
            std_out('No data to post, ignoring', 'ERROR')
            return False

        if 'SC_BEARER' not in environ:
            std_out('Cannot post without Auth Bearer', 'ERROR')
            return False

        if 'SC_ADMIN_BEARER' in environ:
            std_out('Using admin Bearer')
            bearer = environ['SC_ADMIN_BEARER']
        else:
            bearer = environ['SC_BEARER']

        headers = {'Authorization':'Bearer ' + bearer, 'Content-type': 'application/json'}
        post_ok = True

        for sensor_id in self.sensors:
            df = DataFrame(self.data[self.sensors[sensor]]).copy()
            post_ok &= self.post_data_to_device(df, clean_na = clean_na, chunk_size = chunk_size)

        return post_ok

    def post_data_to_device(self, df, clean_na = 'drop', chunk_size = 500, dry_run = False):
        '''
            POST external pandas.DataFrame to the SmartCitizen API
            Parameters
            ----------
                df: pandas DataFrame
                    Contains data in a DataFrame format.
                    Data is posted using the column names of the dataframe
                    Data is posted in UTC TZ so dataframe needs to have located
                    timestamp
                clean_na: string, optional
                    'drop'
                    'drop', 'fill'
                chunk_size: integer
                    chunk size to split resulting pandas DataFrame for posting readings
                dry_run: boolean
                    False
                    Post the payload to the API or just return it
            Returns
            -------
                True if the data was posted succesfully
        '''
        if 'SC_BEARER' not in environ:
            std_out('Cannot post without Auth Bearer', 'ERROR')
            return False

        if 'SC_ADMIN_BEARER' in environ:
            std_out('Using admin Bearer')
            bearer = environ['SC_ADMIN_BEARER']
        else:
            bearer = environ['SC_BEARER']

        headers = {'Authorization':'Bearer ' + bearer, 'Content-type': 'application/json'}

        # Clean df of nans
        df = clean(df, clean_na, how = 'all')
        df.index.name = 'recorded_at'

        # Split the dataframe in chunks
        std_out(f'Splitting post in chunks of size {chunk_size}')
        chunked_dfs = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]

        for i in trange(len(chunked_dfs), file=sys.stdout,
                        desc=f"Posting data for {self.id}..."):

            chunk = chunked_dfs[i].copy()

            # Prepare json post
            payload = {"data":[]}
            for item in chunk.index:
                payload["data"].append(
                    {
                        "recorded_at": localise_date(item, 'UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "sensors": [{
                            "id": column,
                            "value": chunk.loc[item, column]
                        } for column in chunk.columns if not isnan(chunk.loc[item, column])]
                    }
                )

            if dry_run:
                std_out(f'Dry run request to: {self.API_BASE_URL}{self.id}/readings for chunk ({i+1}/{len(chunked_dfs)})')
                return dumps(payload, indent = 2)

            response = post(f'{self.API_BASE_URL}{self.id}/readings',
                            data = dumps(payload), headers = headers)

            if not(response.status_code == 200 or response.status_code == 201):

                std_out (f'Chunk ({i+1}/{len(chunked_dfs)}) post failed. \
                           API responded {response.status_code}', 'ERROR')
                return False

        return True

    def patch_postprocessing(self, dry_run = False):
        '''
            POST postprocessing info into the device in the SmartCitizen API
            Updates all the post info. Changes need to be made info the keys of the postprocessing outside of here

            # Example postprocessing:
            # {
            #   "blueprint_url": "https://github.com/fablabbcn/smartcitizen-data/blob/master/blueprints/sc_21_station_module.json",
            #   "hardware_url": "https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/master/hardware/SCAS210001.json",
            #   "latest_postprocessing": "2020-10-29T08:35:23Z"
            # }
        '''

        if 'SC_ADMIN_BEARER' not in environ:
            std_out('Cannot post without Admin Auth Bearer', 'ERROR')
            return

        headers = {'Authorization':'Bearer ' + environ['SC_ADMIN_BEARER'],
                   'Content-type': 'application/json'}

        post = {"postprocessing_attributes": self.postprocessing}
        post_json = dumps(post)

        if dry_run:
            std_out(f'Dry run request to: {self.API_BASE_URL}{self.id}/')
            return dumps(post_json, indent = 2)

        std_out(f'Posting postprocessing_attributes:\n {post_json}')
        response = patch(f'{self.API_BASE_URL}{self.id}/',
                         data = post_json, headers = headers)

        if response.status_code == 200 or response.status_code == 201:
            std_out(f"Postprocessing posted", "SUCCESS")
            return True
        else:
            std_out(f"API responded with {response.status_code}")

        return False

class MuvApiDevice:

    API_BASE_URL='https://data.waag.org/api/muv/'

    def __init__ (self, did):
        self.id = did
        self.timezone = None
        self.data = None
        self.sensors = None

    def get_device_timezone(self):
        self.timezone = 'Europe/Madrid'
        return self.timezone

    def get_device_sensors(self):
        if self.sensors is None:
            self.sensors = dict()
            for key in config.blueprints:
                if 'muv' not in key: continue
                if 'sensors' in config.blueprints[key]:
                    for sensor_name in config.blueprints[key]['sensors'].keys():
                        # IDs are unique
                        self.sensors[config.blueprints[key]['sensors'][sensor_name]['id']] = sensor_name
        return self.sensors

    def get_device_data(self, min_date = None, max_date = None, frequency = '3Min', clean_na = None):

        if min_date is not None: days_ago = (to_datetime(date.today())-to_datetime(min_date)).days
        else: days_ago = 365 # One year of data

        std_out(f'Requesting data from MUV API')
        std_out(f'Device ID: {self.id}')
        self.get_device_timezone()
        self.get_device_sensors()

        # Get devices
        try:
            if days_ago == -1: url = f'{self.API_BASE_URL}getSensorData?sensor_id={self.id}'
            else: url = f'{self.API_BASE_URL}getSensorData?sensor_id={self.id}&days={days_ago}'
            df = DataFrame(get(url).json())
        except:
            print_exc()
            std_out('Failed sensor request request. Probably no connection', 'ERROR')
            pass
            return None

        try:
            # Rename columns
            df.rename(columns = self.sensors, inplace = True)
            df = df.set_index('time')

            df.index = localise_date(df.index, self.timezone)
            df = df[~df.index.duplicated(keep='first')]
            # Drop unnecessary columns
            df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
            df.drop('id', axis=1, inplace=True)
            # Check for weird things in the data
            df = df.apply(to_numeric, errors='coerce')
            # # Resample
            df = df.resample(frequency).mean()
            df = df.reindex(df.index.rename('TIME'))

            df = clean(df, clean_na, how = 'all')

            self.data = df

        except:
            print_exc()
            std_out('Problem closing up the API dataframe', 'ERROR')
            pass
            return None

        std_out(f'Device {self.id} loaded successfully from API', 'SUCCESS')
        return self.data

class DadesObertesApiDevice:

    API_BASE_URL="https://analisi.transparenciacatalunya.cat/resource/uy6k-2s8r.csv?"

    def __init__ (self, did = None, within = None):
        if did is None and within is None:
            std_out('Specify either station id (=codi_eoi) or within (=(lat, long, radius_meters))')
            return

        if did is not None: self.id = did
        if within is not None: self.id = self.get_id_from_within(within)

        self.timezone = None
        self.data = None
        self.sensors = None
        self.devicejson = None
        self.lat = None
        self.long = None
        self.timezone = None

    @staticmethod
    def get_world_map(city = None, within = None, station_type = None, area_type = None):
        """
        Gets devices from Dades Obertes API with certain requirements
        Parameters
        ----------
            city: string, optional
                Empty string
                City
            within: tuple
                Empty tuple
                Gets the devices within a circle center on lat, long with a radius_meters
                within = tuple(lat, long, radius_meters)
            station_type: string
                None
                Type of station, to choose from: 'background', nan or 'traffic'
            area_type: string
                None
                Type of area, to choose from:  nan, 'peri-urban', 'rural', 'suburban', 'urban'
        Returns
        -------
            A list of eoi codes that comply with the requirements. If no requirements are set, returns all of them
        """
        def is_within_circle(x, within):
            if isnan(x['latitud']): return False
            if isnan(x['longitud']): return False

            return distance(location_A=(within[0], within[1]), location_B=(x['latitude'], x['longitude'])).m < within[2]

        world_map = get("https://analisi.transparenciacatalunya.cat/resource/uy6k-2s8r.csv?")
        df = read_csv(StringIO(world_map.content.decode('utf-8'))).set_index('codi_eoi')

        # Location
        if city is not None: df=df[(df['municipi']==city)]
        if within is not None:

            df['within'] = df.apply(lambda x: is_within_circle(x, within), axis=1)
            df=df[(df['within']==True)]

        # Station type
        if station_type is not None: df=df[(df['tipus_est']==station_type)]
        # Area type
        if area_type is not None: df=df[(df['rea_urb']==area_type)]

        return list(set(list(df.index)))

    def get_id_from_within(self, within):
        '''
            Gets the stations within a radius in meters.
            within = tuple(lat, long, radius_meters)
        '''
        request = self.API_BASE_URL
        request += f'$where=within_circle(geocoded_column,{within[0]},{within[1]},{within[2]})'

        try:
            s = get(request)
        except:
            std_out('Problem with request from API', 'ERROR')
            return None

        if s.status_code == 200 or s.status_code == 201:
            df = read_csv(StringIO(s.content.decode('utf-8')))
        else:
            std_out('API reported {}'.format(s.status_code), 'ERROR')
            return None

        if 'codi_eoi' in df.columns:
            ids = list(set(df.codi_eoi.values))
            if ids == []: std_out('No stations within range', 'ERROR')
            elif len(ids) > 1:
                for ptid in ids:
                    municipi = next(iter(set(df[df.codi_eoi==ptid].municipi.values)))
                    nom_estaci = next(iter(set(df[df.codi_eoi==ptid].nom_estaci.values)))
                    rea_urb = next(iter(set(df[df.codi_eoi==ptid].rea_urb.values)))
                    tipus_est = next(iter(set(df[df.codi_eoi==ptid].tipus_est.values)))

                    std_out(f'{ids.index(ptid)+1} /- {ptid} --- {municipi} - {nom_estaci} - Type: {rea_urb} - {tipus_est}')

                wptid = int(input('Multiple stations found, please select one: ')) - 1

                devid = ids[wptid]
                std_out(f'Selected station in {next(iter(set(df[df.codi_eoi==devid].municipi.values)))} with codi_eoi={devid}')
            else:
                devid = ids[0]
                municipi = next(iter(set(df[df.codi_eoi==devid].municipi.values)))
                nom_estaci = next(iter(set(df[df.codi_eoi==devid].nom_estaci.values)))
                rea_urb = next(iter(set(df[df.codi_eoi==devid].rea_urb.values)))
                tipus_est = next(iter(set(df[df.codi_eoi==devid].tipus_est.values)))
                std_out(f'Found station in {next(iter(set(df[df.codi_eoi==devid].municipi.values)))} with codi_eoi={devid}')
                std_out(f'Found station in {municipi} - {nom_estaci} - {devid} - Type: {rea_urb} - {tipus_est}')

        else:
            std_out('Data is empty', 'ERROR')
            return None

        return devid

    def get_device_sensors(self):

        if self.sensors is None:
            if self.get_device_json() is not None:
                # Get available sensors
                sensors = list(set(self.devicejson.contaminant))

                # Put the ids and the names in lists
                self.sensors = dict()
                for sensor in sensors:
                    for key in config.blueprints:
                        if not search("csic_station",key): continue
                        if 'sensors' in config.blueprints[key]:
                            for sensor_name in config.blueprints[key]['sensors'].keys():
                                if config.blueprints[key]['sensors'][sensor_name]['id'] == str(sensor):
                                    # IDs are unique
                                    self.sensors[sensor] = sensor_name

        return self.sensors

    def get_device_json(self):

        if self.devicejson is None:
            try:
                s = get(self.API_BASE_URL + f'codi_eoi={self.id}')
                if s.status_code == 200 or s.status_code == 201:
                    self.devicejson = read_csv(StringIO(s.content.decode('utf-8')))
                else:
                    std_out('API reported {}'.format(s.status_code), 'ERROR')
            except:
                std_out('Failed request. Probably no connection', 'ERROR')
                pass

        return self.devicejson

    def get_device_timezone(self):

        if self.timezone is None:
            latitude, longitude = self.get_device_lat_long()
            # Localize it
            self.timezone = tz_where.tzNameAt(latitude, longitude)

        std_out ('Device {} timezone is {}'.format(self.id, self.timezone))

        return self.timezone

    def get_device_lat_long(self):

        if self.lat is None or self.long is None:
            if self.get_device_json() is not None:
                latitude = longitude = None
                if 'latitud' in self.devicejson.columns:
                    latitude = next(iter(set(self.devicejson.latitud)))
                    longitude = next(iter(set(self.devicejson.longitud)))

                self.lat = latitude
                self.long = longitude

            std_out ('Device {} is located at {}, {}'.format(self.id, latitude, longitude))

        return (self.lat, self.long)

    def get_device_data(self, min_date = None, max_date = None, frequency = '1H', clean_na = None):
        '''
        Based on code snippet from Marc Roig:
        # I2CAT RESEARCH CENTER - BARCELONA - MARC ROIG (marcroig@i2cat.net)
        '''

        std_out(f'Requesting data from Dades Obertes API')
        std_out(f'Device ID: {self.id}')
        self.get_device_sensors()
        self.get_device_timezone()

        request = self.API_BASE_URL
        request += f'codi_eoi={self.id}'

        if min_date is not None and max_date is not None:
            request += "&$where=data between " + to_datetime(min_date).strftime("'%Y-%m-%dT%H:%M:%S'") \
                    + " and " + to_datetime(max_date).strftime("'%Y-%m-%dT%H:%M:%S'")
        elif min_date is not None:
            request += "&$where=data >= " + to_datetime(min_date).strftime("'%Y-%m-%dT%H:%M:%S'")
        elif max_date is not None:
            request += "&$where=data < " + to_datetime(max_date).strftime("'%Y-%m-%dT%H:%M:%S'")

        try:
            s = get(request)
        except:
            print_exc()
            std_out('Problem with sensor data from API', 'ERROR')
            pass
            return None

        if s.status_code == 200 or s.status_code == 201:
            df = read_csv(StringIO(s.content.decode('utf-8')))
        else:
            std_out('API reported {}'.format(s.status_code), 'ERROR')
            pass
            return None

        # Filter columns
        measures = ['h0' + str(i) for i in range(1,10)]
        measures += ['h' + str(i) for i in range(10,25)]
        # validations = ['v0' + str(i) for i in range(1,10)]
        # validations  += ['v' + str(i) for i in range(10,25)]
        new_measures_names = list(range(1,25))

        columns = ['contaminant', 'data'] + measures# + validations
        try:
            df_subset = df[columns]
            df_subset.columns  = ['contaminant', 'date'] + new_measures_names
        except:
            print_exc()
            std_out('Problem while filtering columns', 'Error')
            return None
        else:
            std_out('Successful filtering', 'SUCCESS')

        # Pivot
        try:
            df = DataFrame([])
            for contaminant in self.sensors.keys():
                if contaminant not in df_subset['contaminant'].values:
                    std_out(f'{contaminant} not in columns. Skipping', 'WARNING')
                    continue
                df_temp= df_subset.loc[df_subset['contaminant']==contaminant].drop('contaminant', 1).set_index('date').unstack().reset_index()
                df_temp.columns = ['hours', 'date', contaminant]
                df_temp['date'] = to_datetime(df_temp['date'])
                timestamp_lambda = lambda x: x['date'] + DateOffset(hours=int(x['hours']))
                df_temp['date'] = df_temp.apply( timestamp_lambda, axis=1)
                df_temp = df_temp.set_index('date')
                df[contaminant] = df_temp[contaminant]
        except:
            # print_exc()
            std_out('Problem while filtering columns', 'Error')
            pass
            return None
        else:
            std_out('Successful pivoting', 'SUCCESS')

        df.index = to_datetime(df.index).tz_localize('UTC').tz_convert(self.timezone)
        df.sort_index(inplace=True)

        # Rename
        try:
            df.rename(columns=self.sensors, inplace=True)
        except:
            # print_exc()
            std_out('Problem while renaming columns', 'Error')
            pass
            return None
        else:
            std_out('Successful renaming', 'SUCCESS')

        # Clean
        df = df[~df.index.duplicated(keep='first')]
        # Drop unnecessary columns
        df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
        # Check for weird things in the data
        df = df.apply(to_numeric, errors='coerce')
        # Resample
        df = df.resample(frequency).mean()

        try:
            df = df.reindex(df.index.rename('TIME'))
            df = clean(df, clean_na, how = 'all')
            self.data = df
        except:
            std_out('Problem closing up the API dataframe', 'ERROR')
            pass
            return None

        std_out(f'Device {self.id} loaded successfully from API', 'SUCCESS')
        return self.data

class NiluApiDevice(object):
    """docstring for IflinkDevice"""
    API_BASE_URL='https://sensors.nilu.no/api/'
    API_CONNECTOR='sensors.nilu.no'

    # Docs
    # https://sensors.nilu.no/api/doc#configure-sensor-schema
    # https://sensors.nilu.no/api/doc#push--sensor-data-by-id

    def __init__ (self, did):

        self.id = did
        self.timezone = None
        self.lat = None
        self.long = None
        self.alt = None
        self.data = None
        self.sensors = None
        self.devicejson = None
        self.last_reading_at = None
        self._api_url = self.API_BASE_URL + f'sensors/{self.id}'

    @property
    def api_url(self):
        return self._api_url

    @staticmethod
    def new_device(name, description = '', resolution = '1Min', epsg = config._epsg, enabled = True, location = None, sensors = None, dry_run = False):
        '''
            Configures the device as a new sensor schema.
            This is a one-time configuration and shouldn't be necessary in a recursive way.
            More information at: https://sensors.nilu.no/api/doc#configure-sensor-schema

            Parameters
            ----------
                name: string
                    Device name
                description: string, optional
                    ''
                    sensor description
                resolution: string, optional
                    '1Min'
                    pandas formatted resolution
                epsg: int, optional
                    4326
                    SRS EPSG code. Defaults to 4326 (WGS84). More info https://spatialreference.org/
                enabled: boolean, optional
                    True
                    flag indicating if sensor is enabled for data transfer
                location: dict
                    None
                    sensor location. If sensor is moving (i.e. position is not fixed),
                    then location must explicitly be set to an empty object: {} when configured. Also see this section.
                    location = {
                                'longitude': longitude (double) – sensor east-west position,
                                'latitude': latitude (double) – sensor north-south position,
                                'altitude': altitude (double) – sensor height above sea level
                                }
                sensors: dict()
                    Dictionary containing necessary information of the sensors to be stored. scdata format:
                    {
                        'SHORT_NAME': {
                                        'desc': 'Channel description',
                                        'id': 'sensor SC platform id',
                                        'units': 'sensor_recording_units'
                                    },
                        ...
                    }
                dry_run: boolean
                    False
                    Post the payload to the API or just return it

            Returns
            -------
                If dry_run, a dict containing the payload
                If not, either False in case of error or a
                dictionary containing:
                    sensorid (int) – sensor identifier
                    message (string) – HTTP status text
                    http-status-code (int) – HTTP status code
                    atom (string) – atom URL to sensor
        '''

        API_BASE_URL='https://sensors.nilu.no/api/'
        API_CONNECTOR='nilu'

        if API_CONNECTOR not in config.connectors:
            std_out(f'No connector for {API_CONNECTOR}', 'ERROR')
            return False

        if 'NILU_BEARER' not in environ:
            std_out('Cannot configure without Auth Bearer', 'ERROR')
            return False

        headers = {'Authorization':'Bearer ' + environ['NILU_BEARER'], 'Content-type': 'application/json'}

        if name is None:
            std_out('Need a name to create a new sensor', 'ERROR')
            return False

        std_out (f'Configuring IFLINK device named {name}')

        # Verify inputs
        flag_error = False

        # EPSG int type
        try:
            int(epsg)
        except:
            std_out('Could not convert epsg to int', 'ERROR')
            flag_error = True
            pass

        # Resolution in seconds
        if not flag_error:
            try:
                resolution_seconds = to_timedelta(resolution).seconds
            except:
                std_out('Could not convert resolution to seconds', 'ERROR')
                flag_error = True
                pass

        # Location
        if not flag_error:
            try:
                location['longitude']
                location['latitude']
                location['altitude']
            except KeyError:
                std_out('Need latitude, longitude and altitude in location dict', 'ERROR')
                flag_error = True
                pass

        if flag_error: return False

        # Construct payload
        payload = {
            "name": name,
            "description": description,
            "resolution": resolution_seconds,
            "srs": {
                "epsg": int(epsg)
            },
            "enabled": enabled
        }

        payload['location'] = location

        parameters = []
        components = []

        # Construct
        for sensor in sensors.keys():
            # Check if it's in the configured connectors
            _sid = str(sensors[sensor]['id'])

            if _sid not in config.connectors[API_CONNECTOR]['sensors']:
                if config._strict:
                    std_out(f"Sensor {sensor} not found in connectors list", "ERROR")
                    return False
                std_out(f"Sensor {sensor} not found in connectors list", "WARNING")
                continue

            units = sensors[sensor]['units']

            _pjson = {
                "name": sensor,
                "type": "double",
                "doc": f"{sensors[sensor]['desc']} in {units}"
            }

            _cjson = {
                "componentid": config.connectors[API_CONNECTOR]['sensors'][_sid]['id'],
                "unitid": config.connectors[API_CONNECTOR]['sensors'][_sid]['unitid'],
                "binding-path": f"/{sensor}",
                "level": config.connectors[API_CONNECTOR]['sensors'][_sid]['level']
            }

            parameters.append(_pjson)
            components.append(_cjson)

        # Add timestamp as long
        parameters.append({
            'name': 'date',
            'type': 'long',
            'doc': 'Date of measurement'
            })

        # Add the converter (we need to push as input-format)
        converters = [{
            "input-type": "string",
            "output-type": "StringEpochTime",
            "target-path": "/date",
            "input-args": {
                "input-format": "yyyy-MM-ddTHH:mm:ssZ"
            }
        }]

        mapping = [{
            "name": "Timestamp",
            "target-path": "/date"
        }]

        payload['parameters'] = parameters
        payload['components'] = components
        payload['converters'] = converters
        payload['mapping'] = mapping

        if dry_run:
            std_out(f'Dry run request to: {API_BASE_URL}sensors/configure')
            return dumps(payload, indent = 2)

        response = post(f'{API_BASE_URL}sensors/configure',
                        data = dumps(payload), headers = headers)

        if response.status_code == 200 or response.status_code == 201:
            std_out('Post successful', 'SUCCESS')
            return response.json()
        else:
            std_out(f'{API_BASE_URL} reported {response.status_code}:\n{response.json()}', 'ERROR')
            return False

    def get_device_json(self, update = False):
        '''
            https://sensors.nilu.no/api/doc#get--sensor-by-id
        '''
        if 'NILU_BEARER' in environ:
            std_out('Auth Bearer found, using it', 'SUCCESS')
            headers = {'Authorization':'Bearer ' + environ['NILU_BEARER']}
        else:
            std_out('Cannot request without bearer', 'ERROR')
            return None

        if self.devicejson is None or update:
            try:
                deviceR = get(f'{self.API_BASE_URL}sensors/{self.id}')
                if deviceR.status_code == 429:
                    std_out('API reported {}. Retrying once'.format(deviceR.status_code),
                            'WARNING')
                    sleep(30)
                    deviceR = get(f'{self.API_BASE_URL}sensors/{self.id}', headers = headers)

                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    self.devicejson = deviceR.json()
                else:
                    std_out('API reported {}'.format(deviceR.status_code), 'ERROR')
            except:
                std_out('Failed request. Probably no connection', 'ERROR')
                pass
        return self.devicejson

    def get_device_lat_long(self, update = False):

        if self.lat is None or self.long is None or update:
            if self.get_device_json(update) is not None:
                latidude = longitude = None
                if 'location' in self.devicejson.keys():
                    latitude, longitude = self.devicejson['location']['latitude'], self.devicejson['location']['longitude']

                self.lat = latitude
                self.long = longitude

        std_out ('Device {} is located at {}, {}'.format(self.id, self.lat, self.long))

        return (self.lat, self.long)

    def get_device_last_reading(self, update = False):
        if 'NILU_BEARER' in environ:
            std_out('Auth Bearer found, using it', 'SUCCESS')
            headers = {'Authorization':'Bearer ' + environ['NILU_BEARER']}
        else:
            std_out('Cannot request without bearer', 'ERROR')
            return None

        if self.last_reading_at is None or update:
            try:
                deviceR = get(f'{self.API_BASE_URL}data/id/{self.id}/maxutc', headers = headers)
                if deviceR.status_code == 429:
                    std_out('API reported {}. Retrying once'.format(deviceR.status_code),
                            'WARNING')
                    sleep(30)
                    deviceR = get(f'{self.API_BASE_URL}data/id/{self.id}/maxutc', headers = headers)

                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    last_json = deviceR.json()
                    last_readings = []
                    for item in last_json:
                        if 'timestamp_from_epoch' in item: last_readings.append(item['timestamp_from_epoch'])

                    self.last_reading_at = localise_date(datetime.fromtimestamp(max(list(set(last_readings)))), 'UTC')
                else:
                    std_out(f'API reported {deviceR.status_code}: {deviceR.json()}', 'ERROR')
            except:
                print_exc()
                std_out('Failed request. Probably no connection', 'ERROR')
                pass

        std_out ('Device {} has last reading at {}'.format(self.id, self.last_reading_at))

        return self.last_reading_at

    def get_device_timezone(self, update = False):

        if self.timezone is None or update:
            latitude, longitude = self.get_device_lat_long(update)
            # Localize it
            if latitude is not None and longitude is not None:
                self.timezone = tz_where.tzNameAt(latitude, longitude, forceTZ=True)

        std_out ('Device {} timezone is {}'.format(self.id, self.timezone))

        return self.timezone

    def get_device_sensors(self, update = False):

        if self.sensors is None or update:
            if self.get_device_json(update) is not None:
                # Get available sensors
                sensors = self.devicejson['components']
                # Put the ids and the names in lists
                self.sensors = dict()
                for sensor in sensors:
                    self.sensors[sensor['id']] = sensor['binding-path'][1:]

        return self.sensors

    def get_device_data(self, min_date = None, max_date = None, frequency = '1Min', clean_na = None):
        '''
            From
            https://sensors.nilu.no/api/doc#get--data-from-utc-timestamp-by-id
            From-to
            https://sensors.nilu.no/api/doc#get--data-from-utc-timestamp-range-by-id
        '''

        if 'NILU_BEARER' in environ:
            std_out('Auth Bearer found, using it', 'SUCCESS')
            headers = {'Authorization':'Bearer ' + environ['NILU_BEARER']}
        else:
            std_out('Cannot request without bearer', 'ERROR')
            return None

        std_out(f'Requesting data from {self.API_BASE_URL}')
        std_out(f'Device ID: {self.id}')

        # Make sure we have the everything we need beforehand
        self.get_device_sensors()
        self.get_device_timezone()
        self.get_device_last_reading()

        if self.timezone is None:
            std_out('Device does not have timezone set, skipping', 'WARNING')
            return None

        # Check start date and end date
        if min_date is not None:
            min_date = localise_date(to_datetime(min_date), 'UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
            std_out (f'Min Date: {min_date}')
        else:
            std_out(f"No min_date specified, requesting all", 'WARNING')
            min_date = localise_date(to_datetime('2021-01-01'), 'UTC').strftime('%Y-%m-%dT%H:%M:%SZ')

        if max_date is not None:
            max_date = localise_date(to_datetime(max_date), 'UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
            std_out (f'Max Date: {max_date}')
        else:
            std_out(f"No max_date specified")

        # Print stuff
        std_out(f'Device timezone: {self.timezone}')
        if not self.sensors.keys():
            std_out(f'Device is empty')
            return None
        else: std_out(f'Sensor IDs: {list(self.sensors.keys())}')

        df = DataFrame()

        # Request sensor per ID
        request = f'{self.API_BASE_URL}data/id/{self.id}/'

        if min_date is not None: request += f'fromutc/{min_date}/'
        if max_date is not None: request += f'toutc/{max_date}'

        # Make request
        sensor_req = get(request, headers = headers)

        # Retry once in case of 429 after 30s
        if sensor_req.status_code == 429:
            std_out('Too many requests, waiting for 1 more retry', 'WARNING')
            sleep (30)
            sensor_req = get(request, headers = headers)

        df = DataFrame(sensor_req.json()).pivot(index='timestamp_from_epoch', columns='component', values='value')
        df.columns.name = None
        df.index = localise_date(to_datetime(df.index, unit='s'), self.timezone)
        df = df.reindex(df.index.rename('TIME'))

        # Drop unnecessary columns
        df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
        # Check for weird things in the data
        df = df.apply(to_numeric, errors='coerce')
        # Resample
        df = df.resample(frequency).mean()
        df = clean(df, clean_na, how = 'all')

        # Rename columns
        d = {}
        for component in self.devicejson['components']:
            if 'name' in component: d[component['name']]=self.sensors[component['id']]
        df = df.rename(columns=d)

        self.data = df

        std_out(f'Device {self.id} loaded successfully from API', 'SUCCESS')
        return self.data

    def post_data_to_device(self, df, clean_na = 'drop', chunk_size = None, dry_run = False):
        '''
            POST external data in the IFLINK API, following
            https://sensors.nilu.no/api/doc#push--sensor-data-by-id
            Parameters
            ----------
                df: pandas DataFrame
                    Contains data in a DataFrame format.
                    Data is posted using the column name of the dataframe
                    Data is posted in UTC TZ so dataframe needs to have located
                    timestamp
                clean_na: string, optional
                    'drop'
                    'drop', 'fill'
                chunk_size: None (not used?)
                    chunk size to split resulting pandas DataFrame for posting readings
                dry_run: boolean
                    False
                    Post the payload to the API or just return it
            Returns
            -------
                True if the data was posted succesfully
        '''

        if 'NILU_BEARER' not in environ:
            std_out('Cannot post without Auth Bearer', 'ERROR')
            return False

        headers = {'Authorization':'Bearer ' + environ['NILU_BEARER'], 'Content-type': 'application/json'}

        # Clean df of nans
        df = clean(df, clean_na, how = 'all')

        # Split the dataframe in chunks
        std_out(f'Splitting post in chunks of size {chunk_size}')
        # chunked_dfs = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]

        for i in trange(len(df.index), file=sys.stdout,
                        desc=f"Posting data for {self.id}..."):

            # chunk = chunked_dfs[i].copy()
            row = DataFrame(df.loc[df.index[i],:]).T
            # Prepare json post
            payload = {}
            payload['date'] = localise_date(df.index[i], 'UTC').strftime('%Y-%m-%dT%H:%M:%SZ')

            for column in row.columns:
                payload[column] = row.loc[df.index[i], column]

            if dry_run:
                std_out(f'Dry run request to: {self.API_BASE_URL}sensors/{self.id}/inbound')
                return dumps(payload, indent = 2)

            response = post(f'{self.API_BASE_URL}sensors/{self.id}/inbound',
                            data = dumps(payload), headers = headers)

            if not(response.status_code == 200 or response.status_code == 201):

                std_out (f'Chunk ({i+1}/{len(chunked_dfs)}) post failed. \
                           API responded {response.status_code}:\n{response.json()}', 'ERROR')
                return False

        return True
