import requests
from tzwhere import tzwhere
import pandas as pd
import numpy as np

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import yaml
from os import getcwd, walk
from os.path import join

# Define base url
base_url = 'https://api.smartcitizen.me/v0/devices/'
kits_url = 'https://api.smartcitizen.me/v0/kits/'

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

from test_utils import currentSensorNames

def getSensors(directory):
    devices = dict()
    mydir = join(directory, 'sensorData')
    for root, dirs, files in walk(mydir):
        for _file in files:
            if _file.endswith(".yaml"):
                filePath = join(root, _file)
                stream = open(filePath)
                yamlFile = yaml.load(stream)
                devices.update(yamlFile)
    return devices

def getKitID(_device, verbose):

    # Get device
    deviceR = requests.get(base_url + '{}/'.format(_device))
    # If status code OK, retrieve data
    if deviceR.status_code == 200 or deviceR.status_code == 201:
        
        deviceRJSON = deviceR.json()
    
        kitID = deviceRJSON['kit']['id']
        
        if verbose:
            print ('Device {} is has this kit ID {}'.format(_device, kitID))
        
        return kitID
    else:
        return 'None'

def getPlatformSensorID():
    kits = requests.get(kits_url)
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
        print (type(sensors.status_code))
        return 'API reported {}'.format(sensors.status_code)   
    return sensors

def getDeviceData(_device, verbose, frequency, start_date, end_date):

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

    # Get device
    deviceR = requests.get(base_url + '{}/'.format(_device))

    # If status code OK, retrieve data
    if deviceR.status_code == 200 or deviceR.status_code == 201:
        
        deviceRJSON = deviceR.json()

        # Get min and max dates
        toDate = deviceRJSON['last_reading_at']
        fromDate = deviceRJSON['added_at']
        
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
        
        print ('\t Sensor IDs')
        print ('\t',sensor_real_ids)
        # Request sensor ID
        for sensor_id in sensor_real_ids:
            indexDF = list()
            dataDF = list()
            # Request sensor per ID
            sensor_id_r = requests.get(base_url + '{}/readings?from={}&rollup={}&sensor_id={}&to={}'.format(_device, start_date.strftime('%Y-%m-%d'), rollup, sensor_id, end_date.strftime('%Y-%m-%d')))
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
                    df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
                    df.sort_index(inplace=True)
                    df = df.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
                    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)

                # Add it to dataframe for each sensor
                else:
                    
                    if dataDF != []:
                        dfT = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_target_names[sensor_real_ids.index(sensor_id)]])
                        dfT.index = pd.to_datetime(dfT.index).tz_localize('UTC').tz_convert(location)
                        dfT.sort_index(inplace=True)
                        dfT = dfT.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)

                        df = df.combine_first(dfT)
        
        df = df.reindex(df.index.rename('Time'))
        return df, toDate, fromDate, hasAlpha
    else:
        return (deviceR.status_code)

def getDeviceLocation(_device):
    # Get device
    deviceR = requests.get(base_url + '{}/'.format(_device))

    # If status code OK, retrieve data
    if deviceR.status_code == 200 or deviceR.status_code == 201:
        
        deviceRJSON = deviceR.json()
        # Get location
        latitude = deviceRJSON['data']['location']['latitude']
        longitude = deviceRJSON['data']['location']['longitude']
        
        # Localize it
        tz_where = tzwhere.tzwhere()
        location = tz_where.tzNameAt(latitude, longitude)

    return location, latitude, longitude

def getReadingsAPI(devices, frequency, start_date, end_date):
    readingsAPI = dict()
    readingsAPI['devices'] = dict()
    # Get dict with sensor history
    sensorHistory = getSensors(getcwd())

    for device in devices:
        print ('Loading device {}'.format(device))
        data, toDate, fromDate, hasAlpha = getDeviceData(device, True, frequency, start_date, end_date)
        location, _, _ = getDeviceLocation(device)
        readingsAPI['devices'][device] = dict()
        if (type(data) == int) and (not (data == 200 or data == 201)):
            readingsAPI['devices'][device]['valid'] = False
            readingsAPI['devices'][device]['status_code'] = data
            
        else:
            # TODO add other info?

            readingsAPI['devices'][device]['data'] = data
            readingsAPI['devices'][device]['valid'] = True
            readingsAPI['devices'][device]['location'] = location

            if hasAlpha:
                print ('\tDevice ID says it had alphasense sensors, loading them...')
                # retrieve data from API for alphasense
                readingsAPI['devices'][device]['alphasense'] = dict()
                try:
                    readingsAPI['devices'][device]['alphasense'] = sensorHistory[device]['gas_pro_board']
                    print ('\tDevice not in history')
                except:
                    print ('\tDevice not in history')
        
        print ('\tLoading Sensor Done')
    return readingsAPI