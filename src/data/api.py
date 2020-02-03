import requests
from tzwhere import tzwhere
from src.saf import *

API_BASE_URL='https://api.smartcitizen.me/v0/devices/'
API_KITS_URL='https://api.smartcitizen.me/v0/kits/'

class api_device:

    def __init__ (self, device_id, from_device = False, verbose = True):
        self.device_id = device_id
        self.kit_id = None
        self.last_reading_at = None
        self.added_at = None
        self.location = None
        self.lat = None
        self.long = None
        self.data = None
        self.verbose = verbose

        if not from_device: self.saf = saf(verbose)

    def std_out(self, msg, type_message = None, force = False):
        if self.verbose or force: 
            if type_message is None: print(msg) 
            elif type_message == 'SUCCESS': print(colored(msg, 'green'))
            elif type_message == 'WARNING': print(colored(msg, 'yellow')) 
            elif type_message == 'ERROR': print(colored(msg, 'red'))

    def get_kit_ID(self):

        if self.kit_id is None:
            self.std_out(f'Requesting kit ID from API for device {self.device_id}')
            # Get device
            try:
                deviceR = requests.get(API_BASE_URL + '{}/'.format(self.device_id))

                # If status code OK, retrieve data
                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    self.kit_id = deviceR.json()['kit']['id']
                    self.std_out ('Device {} is has this kit ID {}'.format(self.device_id, self.kit_id))
                else:
                    self.std_out('API reported {}'.format(deviceR.status_code), 'ERROR')  
            except:
                self.std_out('Failed request. Probably no connection', 'ERROR')  
                pass

        return self.kit_id

    def get_date_last_reading(self):

        if self.last_reading_at is None:
            self.std_out(f'Requesting last reading from API for device {self.device_id}')
            # Get last reading
            try:
                deviceR = requests.get(API_BASE_URL + '{}/'.format(self.device_id))
                
                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    self.last_reading_at = deviceR.json()['last_reading_at']
                    self.std_out ('Device {} has last reading at {}'.format(self.device_id, self.last_reading_at))
                else:
                    self.std_out('API reported {}'.format(deviceR.status_code), 'ERROR')  
            except:
                self.std_out('Failed request. Probably no connection', 'ERROR')  
                pass

        return self.last_reading_at

    def get_device_location(self):
        if self.location is None:
            self.std_out((f'Requesting location from API for device {self.device_id}'))
            # Get location
            try:
                deviceR = requests.get(API_BASE_URL + '{}/'.format(self.device_id))

                # If status code OK, retrieve data
                if deviceR.status_code == 200 or deviceR.status_code == 201: 
                    latidude = longitude = None
                    longitude = None
                    response_json = deviceR.json()
                    if 'location' in response_json.keys(): latitude, longitude = deviceR.json()['location']['latitude'], deviceR.json()['location']['longitude']
                    elif 'data' in response_json.keys(): 
                        if 'location' in response_json['data'].keys(): latitude, longitude = deviceR.json()['data']['location']['latitude'], deviceR.json()['data']['location']['longitude']
                    
                    # Localize it
                    tz_where = tzwhere.tzwhere()
                    self.location = tz_where.tzNameAt(latitude, longitude)
                    self.std_out ('Device {} is located at {}'.format(self.device_id, self.location))

                else:
                    self.std_out('API reported {}'.format(deviceR.status_code), 'ERROR') 
            except:
                self.std_out('Failed request. Probably no connection', 'ERROR')  
                pass
        
        return self.location

    def get_device_lat_long(self):
        self.std_out((f'Requesting lat-long from API for device {self.device_id}'))
        # Get location
        try:
            deviceR = requests.get(API_BASE_URL + '{}/'.format(self.device_id))

            # If status code OK, retrieve data
            if deviceR.status_code == 200 or deviceR.status_code == 201: 
                latidude = longitude = None
                longitude = None
                response_json = deviceR.json()
                if 'location' in response_json.keys(): latitude, longitude = deviceR.json()['location']['latitude'], deviceR.json()['location']['longitude']
                elif 'data' in response_json.keys(): 
                    if 'location' in response_json['data'].keys(): latitude, longitude = deviceR.json()['data']['location']['latitude'], deviceR.json()['data']['location']['longitude']
                
                # Localize it
                self.lat = latitude
                self.long = longitude
                self.std_out ('Device {} is located at {}-{}'.format(self.device_id, latitude, longitude))

            else:
                self.std_out('API reported {}'.format(deviceR.status_code), 'ERROR') 
        except:
            self.std_out('Failed request. Probably no connection', 'ERROR')  
            
        return (self.lat, self.long)
    
    def get_device_added_at(self):
        if self.added_at is None:
            self.std_out(f'Requesting added at from API for device {self.device_id}')
            # Get last reading
            try:
                deviceR = requests.get(API_BASE_URL + '{}/'.format(self.device_id))
                
                if deviceR.status_code == 200 or deviceR.status_code == 201:
                    self.added_at = deviceR.json()['added_at']
                    self.std_out ('Device {} was added at {}'.format(self.device_id, self.added_at))
                else:
                    self.std_out('API reported {}'.format(deviceR.status_code), 'ERROR')  
            except:
                self.std_out('Failed request. Probably no connection', 'ERROR')  
                pass

        return self.added_at

    def get_device_data(self, start_date = None, end_date = None, frequency = '1Min', clean_na = False, clean_na_method = None, currentSensorNames = None):

        if currentSensorNames is None: currentSensorNames = self.saf.current_names
        
        self.std_out(f'Requesting data from API for device {self.device_id}')
        
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

        for item in frequency_convert_LUT:
            if item[1] == frequency_unit:
                rollup_unit = item[0]

        rollup = rollup_value + rollup_unit
        self.std_out(f'Using rollup: {rollup}')

        # Get devices
        try:
            deviceR = requests.get(API_BASE_URL + '{}/'.format(self.device_id))

            # If status code OK, retrieve data
            if deviceR.status_code == 200 or deviceR.status_code == 201:
                
                deviceRJSON = deviceR.json()
                
                # Get available sensors
                sensors = deviceRJSON['data']['sensors']
                
                # Put the ids and the names in lists
                # TO-DO Redo from here
                sensor_ids = list()
                sensor_real_ids = list()
                sensor_names = list()
                sensor_real_names = list()
                sensor_target_names = list()

                for i in range(len(sensors)):
                    sensor_ids.append(deviceRJSON['data']['sensors'][i]['id'])
                    sensor_names.append(deviceRJSON['data']['sensors'][i]['name'])

                # Renaming list based on firmware's short name
                for sensor_id in sensor_ids:
                    for name in currentSensorNames:
                        try:
                            if int(currentSensorNames[name]['id']) == int(sensor_id):
                                sensor_target_names.append(currentSensorNames[name]['shortTitle'])
                                sensor_real_names.append(sensor_names[sensor_ids.index(sensor_id)])
                                sensor_real_ids.append(sensor_id)
                                break
                        except:
                            pass

                if self.location is None:
                    # Get location
                    latitude = deviceRJSON['data']['location']['latitude']
                    longitude = deviceRJSON['data']['location']['longitude']
                    
                    # Localize it
                    tz_where = tzwhere.tzwhere()
                    location = tz_where.tzNameAt(latitude, longitude)
                else: 
                    location = self.location

                # Get min and max getDateLastReading
                toDate = deviceRJSON['last_reading_at'] 
                fromDate = deviceRJSON['added_at']

                # Check start date
                if start_date is None and fromDate is not None:
                    # start_date = datetime.strptime(fromDate, '%Y-%m-%dT%H:%M:%SZ')
                    start_date = pd.to_datetime(fromDate, format = '%Y-%m-%dT%H:%M:%SZ')
                elif start_date is not None:
                    # start_date = datetime.strftime(start_date, '%Y-%m-%dT%H:%M:%SZ')
                    start_date = pd.to_datetime(start_date, format = '%Y-%m-%dT%H:%M:%SZ')
                if start_date.tzinfo is None: start_date = start_date.tz_localize('UTC').tz_convert(location)
                self.std_out (f'Min Date {start_date}')
                
                # Check end date
                if end_date is None and toDate is not None:
                    end_date = pd.to_datetime(toDate, format = '%Y-%m-%dT%H:%M:%SZ')
                elif end_date is not None:
                    end_date = pd.to_datetime(end_date, format = '%Y-%m-%dT%H:%M:%SZ')
                if end_date.tzinfo is None: end_date = end_date.tz_localize('UTC').tz_convert(location)
                self.std_out (f'Max Date {end_date}')
                if start_date > end_date: self.std_out('Ignoring device dates. Probably SD card device', 'WARNING')
                
                # Print stuff if requested
                self.std_out('Kit ID {}'.format(deviceRJSON['kit']['id']))
                if start_date < end_date: self.std_out('From Date {} to Date {}'.format(start_date, end_date))
                self.std_out('Device located in {}'.format(location))

                if deviceRJSON['kit']['id'] in station_kit_ids: hasAlpha = True
                else: hasAlpha = False
                
                self.std_out(f'Sensor IDs:\n{sensor_real_ids}')
                
                # Request sensor ID
                df = None
                for sensor_id in sensor_real_ids:
                    flag_error = False
                    indexDF = list()
                    dataDF = list()

                    # Request sensor per ID
                    request = API_BASE_URL + '{}/readings?'.format(self.device_id)
                    
                    if start_date is not None:
                        if start_date < end_date: request += 'from={}'.format(start_date)
                        else: request += 'from={}'.format('2001-01-01')
                    else: request += 'from={}'.format('2001-01-01')
                    request += '&rollup={}'.format(rollup)
                    request += '&sensor_id={}'.format(sensor_id)
                    request += '&function=avg'
                    if end_date is not None:
                        if end_date > start_date: request += '&to={}'.format(end_date)
                    
                    # Make request
                    sensor_id_r = requests.get(request)
                    try:
                        sensor_id_rJSON = sensor_id_r.json()
                    except:
                        traceback.print_exc()
                        self.std_out('Problem with json data from API', 'ERROR')
                        flag_error = True
                    
                    if 'readings' not in sensor_id_rJSON.keys(): 
                        self.std_out(f'No readings key in request for sensor: {sensor_id}', 'ERROR')
                        flag_error = True
                    
                    elif sensor_id_rJSON['readings'] == []: 
                        self.std_out(f'No data in request for sensor: {sensor_id}', 'ERROR')
                        flag_error = True

                    if flag_error: continue
                    
                    try:

                        # Put the data in lists
                        for item in sensor_id_rJSON['readings']:
                            indexDF.append(item[0])
                            dataDF.append(item[1])

                        # Create result dataframe for first dataframe
                        if df is None:
                            # print 'getting sensor id # 0 at {}'.format(sensor_id)
                            df = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_target_names[sensor_real_ids.index(sensor_id)]])
                            # df.index = pd.to_datetime(df.index).tz_convert(location)
                            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
                            df.sort_index(inplace=True)
                            df = df[~df.index.duplicated(keep='first')]
                            # Drop unnecessary columns
                            df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
                            # Check for weird things in the data
                            df = df.apply(pd.to_numeric, errors='coerce')
                            # # Resample
                            df = df.resample(frequency, limit = 1).mean()

                        # Add it to dataframe for each sensor
                        else:
                            
                            if dataDF != []:
                                
                                dfT = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_target_names[sensor_real_ids.index(sensor_id)]])
                                dfT.index = pd.to_datetime(dfT.index).tz_localize('UTC').tz_convert(location)
                                dfT.sort_index(inplace=True)
                                dfT = dfT[~dfT.index.duplicated(keep='first')]
                                # Drop unnecessary columns
                                dfT.drop([i for i in dfT.columns if 'Unnamed' in i], axis=1, inplace=True)
                                # Check for weird things in the data
                                dfT = dfT.apply(pd.to_numeric,errors='coerce')
                                # Resample
                                dfT = dfT.resample(frequency).mean()
                                df = df.combine_first(dfT)
                           
                    except:
                        traceback.print_exc()
                        self.std_out('Problem with sensor data from API', 'ERROR')
                        flag_error = True
                        pass

                    if flag_error: continue

                try:
                    df = df.reindex(df.index.rename('Time'))
                    
                    if clean_na:
                        if clean_na_method == 'drop':
                            # self.std_out('Cleaning na with drop')
                            df.dropna(axis = 0, how='all', inplace=True)
                        elif clean_na_method == 'fill':
                            df = df.fillna(method='bfill').fillna(method='ffill')
                            # self.std_out('Cleaning na with fill')
                    self.data = df
                    
                except:
                    self.std_out('Problem closing up the API dataframe', 'ERROR')
                    traceback.print_exc()

            else:
                self.std_out('API reported {}'.format(deviceR.status_code), 'ERROR')
        except:
            traceback.print_exc()
            self.std_out('Failed sensor request request. Probably no connection', 'ERROR')
        else:
            self.std_out(f'Device {self.device_id} loaded successfully from API', 'SUCCESS')

        return self.data