from src.saf import std_out
from src.saf import CURRENT_NAMES, FREQ_CONV_LUT, BLUEPRINTS
import pandas as pd
from traceback import print_exc
import requests as req
from tzwhere import tzwhere
from re import search

class sc_api_device:

    API_BASE_URL='https://api.smartcitizen.me/v0/devices/'

    def __init__ (self, id):
        self.id = id
        self.kit_id = None
        self.mac = None
        self.last_reading_at = None
        self.added_at = None
        self.location = None
        self.lat = None
        self.long = None
        self.data = None
        self.sensors = None
        self.devicejson = None

    def get_mac(self):
        if self.mac is None:
            std_out(f'Requesting MAC from API for device {self.id}')
            # Get device
            try:
                deviceR = req.get(self.API_BASE_URL + '{}/'.format(self.id))

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

    def get_device_json(self):
        if self.devicejson is None:
            try:
                deviceR = req.get(self.API_BASE_URL + '{}/'.format(self.id))
                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    self.devicejson = deviceR.json()
                else: 
                    std_out('API reported {}'.format(deviceR.status_code), 'ERROR')  
            except:
                std_out('Failed request. Probably no connection', 'ERROR')  
                pass                
        return self.devicejson

    def get_kit_ID(self):

        if self.kit_id is None:
            if self.get_device_json() is not None:
                self.kit_id = self.devicejson['kit']['id']                
        
        return self.kit_id

    def get_device_last_reading(self):

        if self.last_reading_at is None:
            if self.get_device_json() is not None:
                self.last_reading_at = self.devicejson['last_reading_at']                
        
        std_out ('Device {} has last reading at {}'.format(self.id, self.last_reading_at))

        return self.last_reading_at

    def get_device_location(self):

        if self.location is None:
            if self.get_device_json() is not None:
                latidude = longitude = None
                if 'location' in self.devicejson.keys(): latitude, longitude = self.devicejson['location']['latitude'], self.devicejson['location']['longitude']
                elif 'data' in self.devicejson.keys(): 
                    if 'location' in self.devicejson['data'].keys(): latitude, longitude = self.devicejson['data']['location']['latitude'], self.devicejson['data']['location']['longitude']
                
                # Localize it
                tz_where = tzwhere.tzwhere()
                self.location = tz_where.tzNameAt(latitude, longitude)
        std_out ('Device {} is located at {}'.format(self.id, self.location))               
        
        return self.location

    def get_device_lat_long(self):

        if self.lat is None or self.long is None:
            if self.get_device_json() is not None:
                latidude = longitude = None
                if 'location' in self.devicejson.keys(): latitude, longitude = self.devicejson['location']['latitude'], self.devicejson['location']['longitude']
                elif 'data' in self.devicejson.keys(): 
                    if 'location' in self.devicejson['data'].keys(): latitude, longitude = self.devicejson['data']['location']['latitude'], self.devicejson['data']['location']['longitude']
                
                self.lat = latitude
                self.long = longitude
        
        std_out ('Device {} is located at {}-{}'.format(self.id, latitude, longitude))        
        
        return (self.lat, self.long)
    
    def get_device_added_at(self):

        if self.added_at is None:
            if self.get_device_json() is not None:
                self.added_at = self.devicejson['added_at']                
        
        std_out ('Device {} was added at {}'.format(self.id, self.added_at))

        return self.added_at

    def get_device_sensors(self):

        if self.sensors is None:
            if self.get_device_json() is not None:
                # Get available sensors
                sensors = self.devicejson['data']['sensors']
            
                # Put the ids and the names in lists
                self.sensors = dict()
                for sensor in sensors: 
                    for key in BLUEPRINTS:
                        if not search("sc[k|_]",key): continue
                        if 'sensors' in BLUEPRINTS[key]:
                            for sensor_name in BLUEPRINTS[key]['sensors'].keys(): 
                                if BLUEPRINTS[key]['sensors'][sensor_name]['id'] == str(sensor['id']): 
                                    # IDs are unique
                                    self.sensors[sensor['id']] = sensor_name
        
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

        for item in FREQ_CONV_LUT:
            if item[1] == frequency_unit: 
                rollup_unit = item[0]
                break

        rollup = rollup_value + rollup_unit
        return rollup

    def get_device_data(self, start_date = None, end_date = None, frequency = '1Min', clean_na = None):

        std_out(f'Requesting data from SC API')
        std_out(f'Device ID: {self.id}')

        rollup = self.convert_rollup(frequency)
        std_out(f'Using rollup: {rollup}')

        # Make sure we have the everything we need beforehand
        self.get_device_sensors()
        self.get_device_location()
        self.get_device_last_reading()
        self.get_device_added_at()
        self.get_kit_ID()

        # Check start date
        if start_date is None and self.added_at is not None:
            start_date = pd.to_datetime(self.added_at, format = '%Y-%m-%dT%H:%M:%SZ')
        elif start_date is not None:
            start_date = pd.to_datetime(start_date, format = '%Y-%m-%dT%H:%M:%SZ')
        if start_date.tzinfo is None: start_date = start_date.tz_localize('UTC').tz_convert(self.location)
        std_out (f'Min Date: {start_date}')
        
        # Check end date
        if end_date is None and self.last_reading_at is not None:
            end_date = pd.to_datetime(self.last_reading_at, format = '%Y-%m-%dT%H:%M:%SZ')
        elif end_date is not None:
            end_date = pd.to_datetime(end_date, format = '%Y-%m-%dT%H:%M:%SZ')
        if end_date.tzinfo is None: end_date = end_date.tz_localize('UTC').tz_convert(self.location)
        std_out (f'Max Date: {end_date}')
        if start_date > end_date: std_out('Ignoring device dates. Probably SD card device', 'WARNING')
        
        # Print stuff
        std_out('Kit ID: {}'.format(self.kit_id))
        if start_date < end_date: std_out(f'Dates: from: {start_date}, to: {end_date}')
        std_out(f'Device timezone: {self.location}')
        std_out(f'Sensor IDs:\n{self.sensors.keys()}')

        df = pd.DataFrame()
        
        # Get devices in the sensor first
        for sensor_id in self.sensors.keys(): 

            # Request sensor per ID
            request = self.API_BASE_URL + '{}/readings?'.format(self.id)
            
            request += 'from={}'.format('2001-01-01')
            if start_date is not None:
                if start_date < end_date: request += 'from={}'.format(start_date)
            request += '&rollup={}'.format(rollup)
            request += '&sensor_id={}'.format(sensor_id)
            request += '&function=avg'
            if end_date is not None:
                if end_date > start_date: request += '&to={}'.format(end_date)
            
            # Make request
            sensor_req = req.get(request)
            flag_error = False
            try:
                sensorjson = sensor_req.json()
            except:
                print_exc()
                std_out('Problem with json data from API', 'ERROR')
                flag_error = True
                pass
            
            if 'readings' not in sensorjson.keys(): 
                std_out(f'No readings key in request for sensor: {sensor_id}', 'ERROR')
                flag_error = True
            
            elif sensorjson['readings'] == []: 
                std_out(f'No data in request for sensor: {sensor_id}', 'WARNING')
                flag_error = True

            if flag_error: continue

            # Put 
            try:
                dfsensor = pd.DataFrame(sensorjson['readings']).set_index(0)
                dfsensor.columns = [self.sensors[sensor_id]]
                dfsensor.index = pd.to_datetime(dfsensor.index).tz_localize('UTC').tz_convert(self.location)
                dfsensor.sort_index(inplace=True)
                dfsensor = dfsensor[~dfsensor.index.duplicated(keep='first')]
                
                # Drop unnecessary columns
                dfsensor.drop([i for i in dfsensor.columns if 'Unnamed' in i], axis=1, inplace=True)
                # Check for weird things in the data
                dfsensor = dfsensor.apply(pd.to_numeric, errors='coerce')
                # Resample
                dfsensor = dfsensor.resample(frequency, limit = 1).mean()

                df = df.combine_first(dfsensor)
            except:
                print_exc()
                std_out('Problem with sensor data from API', 'ERROR')
                flag_error = True
                pass                

            if flag_error: continue

            try:
                df = df.reindex(df.index.rename('Time'))
                
                if clean_na is not None:
                    if clean_na == 'drop':
                        # std_out('Cleaning na with drop')
                        df.dropna(axis = 0, how='all', inplace=True)
                    elif clean_na == 'fill':
                        df = df.fillna(method='bfill').fillna(method='ffill')
                        # std_out('Cleaning na with fill')
                self.data = df
                
            except:
                std_out('Problem closing up the API dataframe', 'ERROR')
                print_exc()

        std_out(f'Device {self.id} loaded successfully from API', 'SUCCESS')
        return self.data

class muv_api_device:

    API_BASE_URL='https://data.waag.org/api/muv/'

    def __init__ (self, id):
        self.id = id
        self.location = None
        self.data = None
        self.sensors = None

    def get_device_location(self):
        self.location = 'Europe/Madrid'
        return self.location

    def get_device_sensors(self):
        if self.sensors is None:
            self.sensors = dict()
            for key in BLUEPRINTS:
                if 'muv' not in key: continue
                if 'sensors' in BLUEPRINTS[key]:
                    for sensor_name in BLUEPRINTS[key]['sensors'].keys():
                        # IDs are unique
                        self.sensors[BLUEPRINTS[key]['sensors'][sensor_name]['id']] = sensor_name
        return self.sensors

    def get_device_data(self, start_date = None, end_date = None, frequency = '1Min', clean_na = None):

        std_out(f'Requesting data from MUV API')
        std_out(f'Device ID: {self.id}')
        self.get_device_location()
        self.get_device_sensors()        
        
        # Get devices
        try:
            url = self.API_BASE_URL + 'getSensorData?sensor_id={}/'.format(self.id)
            df = pd.read_json(url)
        except:
            print_exc()
            std_out('Failed sensor request request. Probably no connection', 'ERROR')
        else:
            std_out(f'Device {self.id} loaded successfully from API', 'SUCCESS')

        try:
            # TODO check
            df.rename(columns=self.sensors, inplace=True)
            df = df.set_index('Time')
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(self.location)
            df = df[~df.index.duplicated(keep='first')]
            # Drop unnecessary columns
            df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
            # Check for weird things in the data
            df = df.apply(pd.to_numeric, errors='coerce')
            # # Resample
            df = df.resample(frequency, limit = 1).mean()

            df = df.reindex(df.index.rename('Time'))
                
            if clean_na is not None:
                if clean_na == 'drop':
                    df.dropna(axis = 0, how='all', inplace=True)
                elif clean_na == 'fill':
                    df = df.fillna(method='bfill').fillna(method='ffill')
            self.data = df
                
        except:
            std_out('Problem closing up the API dataframe', 'ERROR')
            print_exc()

        return self.data