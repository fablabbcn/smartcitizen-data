import requests
from tzwhere import tzwhere
import pandas as pd
import numpy as np

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

# Define base url
base_url = 'https://api.smartcitizen.me/v0/devices/'
rollup = '5m' # https://developer.smartcitizen.me/#get-historical-readings

# Get this automatically
station_kit_id = 21
kit_kit_id = 11

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

def getDeviceData(_device, verbose, frequency):

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
        sensor_names = list()
        for i in range(len(sensors)):
            sensor_ids.append(deviceRJSON['data']['sensors'][i]['id'])
            sensor_names.append(deviceRJSON['data']['sensors'][i]['name'])
        
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
            # print sensor_ids
            # print sensor_names

        if deviceRJSON['kit']['id'] == station_kit_id:
            hasAlpha = True
        else:
            hasAlpha = False
        
        # Request sensor ID
        for sensor_id in sensor_ids:
            
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
                if sensor_ids.index(sensor_id) == 0:
                    # print 'getting sensor id # 0 at {}'.format(sensor_id)
                    df = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_names[sensor_ids.index(sensor_id)]])
                    df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
                    df.sort_index(inplace=True)
                    df = df.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
                    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
                
                # Add it to dataframe for each sensor
                else:
                    # print 'getting sensor id {}'.format(sensor_id)
                    dfT = pd.DataFrame(dataDF, index= indexDF, columns = [sensor_names[sensor_ids.index(sensor_id)]])
                    dfT.index = pd.to_datetime(dfT.index).tz_localize('UTC').tz_convert(location)
                    dfT.sort_index(inplace=True)
                    dfT = dfT.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
                    
                    df = df.combine_first(dfT)
                
        return df, location, toDate, fromDate, hasAlpha
    else:
        return (deviceR.status_code)

def getReadingsAPI(_devices, test_name, frequency):
    readingsAPI = {}
    readingsAPI[test_name] = dict()
    readingsAPI[test_name]['devices'] = dict()
    for device in _devices:
        print 'Loading device {}'.format(device)
        data, location, toDate, fromDate, hasAlpha = getDeviceData(device, True, frequency)
        readingsAPI[test_name]['devices'][device] = dict()
        if (type(data) == int) and (not (data == 200 or data == 201)):
            readingsAPI[test_name]['devices'][device]['valid'] = False
            readingsAPI[test_name]['devices'][device]['status_code'] = data
            
        else:
            # TODO add other info?
            
            readingsAPI[test_name]['devices'][device]['data'] = data
            readingsAPI[test_name]['devices'][device]['valid'] = True
            readingsAPI[test_name]['devices'][device]['location'] = location

            if hasAlpha:
                print 'Device ID says it had alphasense sensors'
                # retrieve data from API for alphasense
                readingsAPI[test_name]['devices'][device]['alphasense'] = dict()
                alphaDelta = dict()
                alphaDelta['CO'] = 'TEMPORARY_CO'
                alphaDelta['NO2'] = 'TEMPORARY_NO2'
                alphaDelta['O3'] = 'TEMPORARY_O3'
                alphaDelta['SLOTS'] = 'TEMPORARY_SLOTS'
                readingsAPI[test_name]['devices'][device]['alphasense'] = alphaDelta
            
    return readingsAPI