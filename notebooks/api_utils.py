import requests
from tzwhere import tzwhere
import pandas as pd
import numpy as np

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

# Define base url
base_url = 'https://api.smartcitizen.me/v0/devices/'
kits_url = 'https://api.smartcitizen.me/v0/kits/'

# TODO: Get this automatically
station_kit_id = 19
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

def getKitID(_device, verbose):
    # Get device
    deviceR = requests.get(base_url + '{}/'.format(_device))
    # If status code OK, retrieve data
    if deviceR.status_code == 200 or deviceR.status_code == 201:
        
        deviceRJSON = deviceR.json()
    
        kitID = deviceRJSON['kit']['id']
        
        if verbose:
            print 'Device {} is has this kit ID {}'.format(_device, kitID)
        
        return kitID
    else:
        print type(deviceR.status_code)
        return 'API reported {}'.format(deviceR.status_code)

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
        print type(sensors.status_code)
        return 'API reported {}'.format(sensors.status_code)   
    return sensors

def getDeviceData(_device, verbose, frequency):

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
    print 'Getting device {} at url {}'.format(_device, base_url + '{}/'.format(_device))

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
                # print 'Current sensor names: {}, type: {}'.format(currentSensorNames[name]['id'], type(currentSensorNames[name]['id']))
                # print 'sensor_id: {}, type: {}'.format(sensor_id, type(sensor_id))
                try:
                    if int(currentSensorNames[name]['id']) == int(sensor_id):
                        # print 'Current sensor names: {}, type: {}'.format(currentSensorNames[name]['id'], type(currentSensorNames[name]['id']))
                        # print 'sensor_id: {}, type: {}'.format(sensor_id, type(sensor_id))
                        # print 'sensor real names: {}'.format(sensor_names[sensor_ids.index(sensor_id)])
                        # print currentSensorNames[name]['shortTitle'] 
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
            print 'Kit ID {}'.format(deviceRJSON['kit']['id'])
            print '\tFrom Date {} to Date {}'.format(fromDate, toDate)
            print '\tDevice located in {}'.format(location)
            # print 'Sensor IDs'
            # for sensor_id in sensor_real_ids:
            #     print 'Index: {}'.format(sensor_real_ids.index(sensor_id))
            #     print 'Sensor ID: {}'.format(sensor_id)
            #     print 'Sensor Name Platform: {}'.format(sensor_real_names[sensor_real_ids.index(sensor_id)])
            #     print 'Sensor Name Target: {}'.format(sensor_target_names[sensor_real_ids.index(sensor_id)])

        if deviceRJSON['kit']['id'] == station_kit_id:
            hasAlpha = True
        else:
            hasAlpha = False
        
        # Request sensor ID
        for sensor_id in sensor_real_ids:
            indexDF = list()
            dataDF = list()
            
            # Request sensor per ID
            sensor_id_r = requests.get(base_url + '{}/readings?from={}&rollup={}&sensor_id={}&to={}'.format(_device, fromDate, rollup, sensor_id, toDate))
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
                    # df.rename(columns={sensor_names[sensor_ids.index(sensor_id)]: sensor_target_names[sensor_ids.index(sensor_id)]}, inplace=True)

                # Add it to dataframe for each sensor
                else:
                    # print 'getting sensor id {}'.format(sensor_id)
                    if dataDF != []:
                        dfT = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_target_names[sensor_real_ids.index(sensor_id)]])
                        dfT.index = pd.to_datetime(dfT.index).tz_localize('UTC').tz_convert(location)
                        dfT.sort_index(inplace=True)
                        dfT = dfT.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
                        # dfT.rename(columns={sensor_names[sensor_ids.index(sensor_id)]: sensor_target_names[sensor_ids.index(sensor_id)]}, inplace=True)

                        df = df.combine_first(dfT)
                
        return df, location, toDate, fromDate, hasAlpha, latitude, longitude 
    else:
        return (deviceR.status_code)

def getReadingsAPI(_devices, frequency):
    readingsAPI = dict()
    readingsAPI['devices'] = dict()
    for device in _devices:
        print 'Loading device {}'.format(device)
        data, location, toDate, fromDate, hasAlpha, latitude, longitude = getDeviceData(device, True, frequency)
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
                print 'Device ID says it had alphasense sensors'
                # retrieve data from API for alphasense
                readingsAPI['devices'][device]['alphasense'] = dict()
                alphaDelta = dict()
                alphaDelta['CO'] = 'TEMPORARY_CO'
                alphaDelta['NO2'] = 'TEMPORARY_NO2'
                alphaDelta['O3'] = 'TEMPORARY_O3'
                alphaDelta['SLOTS'] = 'TEMPORARY_SLOTS'
                readingsAPI['devices'][device]['alphasense'] = alphaDelta
            
    return readingsAPI