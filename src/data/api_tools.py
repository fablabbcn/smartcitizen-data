import requests
from tzwhere import tzwhere
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import yaml
from os import getcwd, walk
from os.path import join

from src.data.variables import *
import traceback
from termcolor import colored 

# TODO: Get this automatically
station_kit_ids = (19, 21)
kit_kit_id = 11

# Convertion table from API SC to Pandas
# https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
# https://developer.smartcitizen.me/#get-historical-readings
frequencyConvertLUT = (['y','A'],
    ['M','M'],
    ['w','W'],
    ['d','D'],
    ['h','H'],
    ['m','Min'],
    ['s','S'],
    ['ms','ms'])

def getSensors(mydir):
    devices = dict()
    for root, dirs, files in walk(mydir):
        for _file in files:
            if _file.endswith(".yaml"):
                filePath = join(root, _file)
                stream = open(filePath)
                yamlFile = yaml.load(stream)
                devices.update(yamlFile)
    return devices

def getKitID(_device, verbose = True):
    try:
        # Get device
        deviceR = requests.get(API_BASE_URL + '{}/'.format(_device))

        # If status code OK, retrieve data
        if deviceR.status_code == 200 or deviceR.status_code == 201:
            
            deviceRJSON = deviceR.json()
            kitID = deviceRJSON['kit']['id']
            
            if verbose: print ('Device {} is has this kit ID {}'.format(_device, kitID))
            
            return kitID

        else:
            return None
    except:
        print(colored('Failed kit ID request. Probably no connection', 'red'))  
        return None

def getPlatformSensorID(verbose = True):
    try:
        kits = requests.get(API_KITS_URL)
        # If status code OK, retrieve data
        if kits.status_code == 200 or kits.status_code == 201:
            
            kitsJSON = kits.json()
            sensorsDict ={}
            
            for kit in kitsJSON:
                for sensor in kit['sensors']:
                    ID = sensor['id']
                    if not ID in sensorsDict.keys():
                        sensorsDict[ID] = dict()
                        sensorsDict[ID]['name'] = sensor['name']
                        sensorsDict[ID]['unit'] = sensor['unit']
            
            return sensorsDict
        else:
            print(colored('API reported {}'.format(sensors.status_code), 'red')) 
            return None
        return sensors
    except:
        print(colored('Failed Sensor request. Probably no connection', 'red'))  
        return None

def getDateLastReading(_device, verbose = True):
    # Get device
    try:
        deviceR = requests.get(API_BASE_URL + '{}/'.format(_device))
        if verbose: print (f'Requesting last reading from API for device {_device}')
        if deviceR.status_code == 200 or deviceR.status_code == 201:
            return deviceR.json()['last_reading_at']
        else:
            print(colored('API reported {}'.format(deviceR.status_code), 'red'))  
            return None
    except:
        print(colored('Failed Date request. Probably no connection', 'red'))  
        return None


def getDeviceData(_device, verbose, frequency, start_date, end_date, location, currentSensorNames, clean_na, clean_na_method):
    if verbose: print (f'Requesting data from API for device {_device}')
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

    for item in frequencyConvertLUT:
        if item[1] == frequency_unit:
            rollup_unit = item[0]

    rollup = rollup_value + rollup_unit
    if verbose: print (f'Using rollup: {rollup}')

    try:
        # Get device
        deviceR = requests.get(API_BASE_URL + '{}/'.format(_device))

        # If status code OK, retrieve data
        if deviceR.status_code == 200 or deviceR.status_code == 201:
            
            deviceRJSON = deviceR.json()

            # Get min and max getDateLastReading
            toDate = deviceRJSON['last_reading_at'] 
            fromDate = deviceRJSON['added_at']
            if start_date is None and fromDate is not None:
                start_date = datetime.strptime(fromDate, '%Y-%m-%dT%H:%M:%SZ')
            elif start_date is not None:
                start_date = datetime.strftime(start_date, '%Y-%m-%dT%H:%M:%SZ')
            if verbose: print ('Min Date', start_date)

            if end_date is None and toDate is not None:
                end_date = datetime.strptime(toDate, '%Y-%m-%dT%H:%M:%SZ')
            elif end_date is not None:
                try:
                    end_date = datetime.strftime(end_date, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    pass
            if verbose: print ('Max Date', end_date)
            
            # Get available sensors
            sensors = deviceRJSON['data']['sensors']
            
            # Put the ids and the names in lists
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

            if location is None:
                # Get location
                latitude = deviceRJSON['data']['location']['latitude']
                longitude = deviceRJSON['data']['location']['longitude']
                
                # Localize it
                tz_where = tzwhere.tzwhere()
                location = tz_where.tzNameAt(latitude, longitude)
            
            # Print stuff if requested
            if verbose:
                print ('Kit ID {}'.format(deviceRJSON['kit']['id']))
                print ('\tFrom Date {} to Date {}'.format(start_date, end_date))
                print ('\tDevice located in {}'.format(location))

            if deviceRJSON['kit']['id'] in station_kit_ids:
                hasAlpha = True
            else:
                hasAlpha = False
            
            if verbose: print (f'\tSensor IDs:\n\t{sensor_real_ids}')
            
            # Request sensor ID
            for sensor_id in sensor_real_ids:
                indexDF = list()
                dataDF = list()

                # Request sensor per ID
                request = API_BASE_URL + '{}/readings?'.format(_device)
                if start_date is not None: request += 'from={}'.format(start_date)
                request += '&rollup={}'.format(rollup)
                request += '&sensor_id={}'.format(sensor_id)
                request += '&function=avg'
                
                if end_date is not None: request += '&to={}'.format(end_date)
                # Make request
                sensor_id_r = requests.get(request)
                
                try:
                    sensor_id_rJSON = sensor_id_r.json()

                    # Put the data in lists
                    if 'readings' in sensor_id_rJSON:
                        for item in sensor_id_rJSON['readings']:
                            indexDF.append(item[0])
                            dataDF.append(item[1])

                        # Create result dataframe for first dataframe
                        if sensor_real_ids.index(sensor_id) == 0:
                            # print 'getting sensor id # 0 at {}'.format(sensor_id)
                            df = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_target_names[sensor_real_ids.index(sensor_id)]])
                            # df.index = pd.to_datetime(df.index).tz_convert(location)
                            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
                            df.sort_index(inplace=True)
                            df = df[~df.index.duplicated(keep='first')]
                            # Drop unnecessary columns
                            df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
                            # Check for weird things in the data
                            df = df.apply(pd.to_numeric,errors='coerce')
                            # # Resample
                            df = df.resample(frequency, limit = 1).mean()
                            # Remove na
                            # if clean_na:
                            #     if clean_na_method == 'fill':
                            #         df = df.fillna(method='bfill', limit=2).fillna(method='ffill', limit=2)

                        # Add it to dataframe for each sensor
                        else:
                            
                            if dataDF != []:
                                dfT = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_target_names[sensor_real_ids.index(sensor_id)]])
                                # dfT.index = pd.to_datetime(dfT.index).tz_convert(location)
                                dfT.index = pd.to_datetime(dfT.index).tz_localize('UTC').tz_convert(location)

                                dfT.sort_index(inplace=True)
                                dfT = dfT[~dfT.index.duplicated(keep='first')]
                                # Drop unnecessary columns
                                dfT.drop([i for i in dfT.columns if 'Unnamed' in i], axis=1, inplace=True)
                                # Check for weird things in the data
                                dfT = dfT.apply(pd.to_numeric,errors='coerce')
                                # Resample
                                dfT = dfT.resample(frequency).mean()
                                # # Remove na
                                # if clean_na:
                                #     if clean_na_method == 'fill':
                                #         dfT = dfT.fillna(method='bfill', limit=2).fillna(method='ffill', limit=2)

                                df = df.combine_first(dfT)
                except:
                    traceback.print_exc()
                    pass
            
            try:
                df = df.reindex(df.index.rename('Time'))
                
                if clean_na:
                    if clean_na_method == 'drop':
                        if verbose: print ('Cleaning na with drop')
                        df.dropna(axis = 0, how='all', inplace=True)
                    elif clean_na_method == 'fill':
                        df = df.fillna(method='bfill').fillna(method='ffill')
                        if verbose: print ('Cleaning na with fill')
                return df, hasAlpha
            except:
                if verbose: print ('Problem closing up the API dataframe')
                traceback.print_exc()
                return None, None

        else:
            print(colored('API reported {}'.format(deviceR.status_code), 'red')) 
            return None, None
    except:
        print(colored('Failed sensor request request. Probably no connection', 'red'))  
        return None, None

def getDeviceLocation(_device, verbose = True):
    if verbose: print (f'Requesting location from API for device {_device}')
    # Get device
    try:
        deviceR = requests.get(API_BASE_URL + '{}/'.format(_device))

        # If status code OK, retrieve data
        if deviceR.status_code == 200 or deviceR.status_code == 201:
            
            deviceRJSON = deviceR.json()
            # Get location
            latitude = deviceRJSON['data']['location']['latitude']
            longitude = deviceRJSON['data']['location']['longitude']
            
            # Localize it
            tz_where = tzwhere.tzwhere()
            location = tz_where.tzNameAt(latitude, longitude)

            return location
        else:
            print(colored('API reported {}'.format(deviceR.status_code), 'red')) 
            return None
    except:
        traceback.print_exc()
        return None

def getReadingsAPI(devices, frequency, start_date, end_date, currentSensorNames, dataDirectory, location = None, clean_na = True, clean_na_method = 'fill', verbose = True):
    readingsAPI = dict()
    readingsAPI['devices'] = dict()
    
    # Get dict with sensor history
    sensorHistory = getSensors(join(dataDirectory, 'interim'))

    for device in devices:
        if verbose: print ('Loading device {} from API'.format(device))
        if location is None: getDeviceLocation(device, verbose)
        data, hasAlpha = getDeviceData(device, verbose, frequency, start_date, end_date, location, currentSensorNames, clean_na, clean_na_method)

        readingsAPI['devices'][device] = dict()
        
        if (type(data) == int) and (not (data == 200 or data == 201)) and (location is not None):

            readingsAPI['devices'][device]['valid'] = False
            readingsAPI['devices'][device]['status_code'] = data
            
        else:

            readingsAPI['devices'][device]['data'] = data
            readingsAPI['devices'][device]['valid'] = True
            readingsAPI['devices'][device]['location'] = location

            if hasAlpha:
                if verbose: print ('Device ID says it had alphasense sensors, loading them...')
                # retrieve data from API for alphasense
                readingsAPI['devices'][device]['alphasense'] = dict()
                try:
                    readingsAPI['devices'][device]['alphasense'] = sensorHistory[device]['gas_pro_board']
                except:
                    if verbose: print ('Device not in history')
        
    if verbose: print ('Loading Sensor Done')
    return readingsAPI