import urllib2
import re
import os
from os.path import dirname, join
import yaml
# import markdown
from tzwhere import tzwhere
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import pandas as pd
import numpy as np

pollutantLUT = (['CO', 28, 'ppm'],
                ['NO', 30, 'ppb'],
                ['NO2', 46, 'ppb'],
                ['O3', 48, 'ppb'])

ref_append = 'REF'

currentSensorsh = ('https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/master/lib/Sensors/Sensors.h')

def getSensorNames(_sensorsh):
    # read only 20 000 chars
    data = urllib2.urlopen(_sensorsh).read(20000)
    # split it into lines
    data = data.split("\n") 
    sensorNames = dict()
    lineSensors = len(data)
    for line in data:
        
        if 'class AllSensors' in line:
            lineSensors = data.index(line)
            
        if data.index(line) > lineSensors:
                
                if 'OneSensor' in line and '{' in line and '}' in line and '/*' not in line:
                    # Split commas
                    lineTokenized =  line.strip('').split(',')
                    # print len(lineTokenized)
                    # Elimminate unnecessary elements
                    lineTokenizedSub = list()
                    for item in lineTokenized:
                            item = re.sub('\t', '', item)
                            item = re.sub('OneSensor', '', item)
                            item = re.sub('{', '', item)
                            item = re.sub('}', '', item)
                            #item = re.sub(' ', '', item)
                            item = re.sub('"', '', item)
                            
                            if item != '': 
                                while item[0] == ' ' and len(item)>0: item = item[1:]
                            lineTokenizedSub.append(item)
                    lineTokenizedSub = lineTokenizedSub[:-1]


                    if len(lineTokenizedSub) > 2:
                            sensorLocation = re.sub(' ', '', lineTokenizedSub[0])
                            sensorID = re.sub(' ','', lineTokenizedSub[1])
                            sensorNames[sensorID] = dict()
                            sensorNames[sensorID]['SensorLocation'] = sensorLocation
                            if len(lineTokenizedSub)>7:
                                    sensorNames[sensorID]['shortTitle'] = re.sub(' ', '', lineTokenizedSub[2])
                                    sensorNames[sensorID]['title'] = lineTokenizedSub[3]
                                    sensorNames[sensorID]['id'] = re.sub(' ', '', lineTokenizedSub[4])
                                    sensorNames[sensorID]['unit'] = lineTokenizedSub[len(lineTokenizedSub)-1]
                            else:
                                    sensorNames[sensorID]['shortTitle'] = lineTokenizedSub[2]
                                    sensorNames[sensorID]['title'] = lineTokenizedSub[2]
                                    sensorNames[sensorID]['id'] = re.sub(' ', '', lineTokenizedSub[3])
                                    sensorNames[sensorID]['unit'] = lineTokenizedSub[len(lineTokenizedSub)-1]
    return sensorNames

currentSensorNames = getSensorNames(currentSensorsh)

def CHANNEL_NAME(_sensorNames, _measurement, _concat1, _concat2, _SensorLocation, _unit):
    sensor_name = ''
    for name in _sensorNames:
        nameLocation = _sensorNames[name]['SensorLocation']
        nameUnit = _sensorNames[name]['unit']
        if _SensorLocation == 'BOARD_AUX':
            if nameLocation == _SensorLocation and '{}{}'.format(_concat1, _concat2) in name and _measurement in name:
                sensor_name = str(_sensorNames[name]['shortTitle'])
                return sensor_name
            elif nameLocation == _SensorLocation and not '{}{}'.format(_concat1, _concat2) in name and '{}'.format(_measurement) in name:
                sensor_name = str(_sensorNames[name]['shortTitle'])
    
        elif _SensorLocation == 'BOARD_URBAN':
            if nameLocation == _SensorLocation and '{}{}'.format(_concat1, _concat2) in name and '{}'.format(_measurement) in name and nameUnit in _unit:
                sensor_name = str(_sensorNames[name]['shortTitle'])
                return sensor_name
    
    return sensor_name

def getTests(directory):
    tests = dict()
    mydir = join(directory, 'data')
    for root, dirs, files in os.walk(mydir):
        for _file in files:
            if _file.endswith(".yaml"):
                filePath = join(root, _file)
                stream = open(filePath)
                yamlFile = yaml.load(stream)
                tests[yamlFile['test']['id']] = root
                #print [yamlFile['test']['id'], filePath]
    return tests

selectedTests = tuple()
def selectTests(x):
    global selectedTests
    selectedTests = list(x)
    
def clearTests():
    clear_output()
    selectedTests = ()

def loadTest(frequency):
    
    # print selectedTests
    readings = {}
    clear_output()
    for testPath in selectedTests:
        # Find Yaml
        filePath = join(testPath, 'test_description.yaml')
        stream = file(filePath)
        test = yaml.load(stream)
        test_id = test['test']['id']
        
        readings[test_id] = dict()
        readings[test_id]['devices'] = dict()
        readings[test_id]['commit_hash'] = test['commit_hash']
        commitSensorsh = ('https://raw.githubusercontent.com/fablabbcn/smartcitizen-kit-20/' + readings[test_id]['commit_hash'] + '/lib/Sensors/Sensors.h')
        commitSensorNames = getSensorNames(commitSensorsh)

        targetSensorNames = list()
        for types in ('W','A'):
            for slot in (1,2,3):
                targetSensorNames.append(CHANNEL_NAME(currentSensorNames, 'GASES', slot, types, 'BOARD_AUX', ''))
        targetSensorNames.append('EXT_TEMP')
        targetSensorNames.append('EXT_HUM')
        targetSensorNames.append('CO_MICS_RAW')
        targetSensorNames.append('NO2_MICS_RAW')
        targetSensorNames.append('TEMP')
        targetSensorNames.append('HUM')
        
        testSensorNames = list()
        for types in ('W','A'):
            for slot in (1,2,3):
                testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'ALPHA', slot, types, 'BOARD_AUX', ''))
        testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'TEMPERATURE', 0, '?ONE', 'BOARD_AUX', ''))
        testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'HUMIDITY', 0, '?ONE', 'BOARD_AUX', ''))
        testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'SENSOR_CO', '', '', 'BOARD_URBAN', 'kOhm'))
        testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'SENSOR_NO2', '', '', 'BOARD_URBAN', 'kOhm'))
        testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'TEMPERATURE', '', '', 'BOARD_URBAN', 'C'))
        testSensorNames.append(CHANNEL_NAME(commitSensorNames, 'HUMIDITY', '', '', 'BOARD_URBAN', '%'))

        # Get test metadata
        test_init_date = test['test']['init_date']
        test_end_date = test['test']['end_date']
        
        print('------------------------------------------------------')
        display(Markdown('## Test Load'))

        display(Markdown('Loading test **{}** performed from {} to {}'.format(test_id, test_init_date, test_end_date)))
    
        display(Markdown('Test performed with commit **{}**'.format(readings[test_id]['commit_hash'])))

        display(Markdown(test['test']['comment']))
        display(Markdown('### KIT'))

        # Open all kits
        for kit in test['test']['devices']['kits']:

            display(Markdown('#### {}'.format(kit)))
            # Get fileName
            fileNameProc = test['test']['devices']['kits'][kit]['fileNameProc']
            # frequency = test['test']['devices']['kits'][kit]['frequency']
            fileData = join(testPath, fileNameProc)
            location = test['test']['devices']['kits'][kit]['location']
            display(Markdown('Kit **{}** located **{}**'.format(kit, location)))
            
            # Create pandas dataframe
            df = pd.read_csv(fileData, verbose=False, skiprows=[1]).set_index('Time')
            # df = pd.read_csv(fileData, verbose=False).set_index('Time')
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
            
            df.sort_index(inplace=True)
            df = df.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
            df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
            
            # Create dictionary and add it to the readings key
            if len(targetSensorNames) == len(testSensorNames) and len(targetSensorNames) > 0:
                for i in range(len(targetSensorNames)):
                    if not (testSensorNames[i] == '') and not (testSensorNames[i] == targetSensorNames[i]) and testSensorNames[i] in df.columns:
                        df.rename(columns={testSensorNames[i]: targetSensorNames[i]}, inplace=True)
                        print ('\tRenaming column _{}_ to _{}_'.format(testSensorNames[i], targetSensorNames[i]))
            
            kitDict = dict()
            kitDict['location'] = location
            kitDict['data'] = df
            
            readings[test_id]['devices'][kit] = kitDict
            
            
            ## Check if it's a STATION and retrieve alphadelta codes
            if test['test']['devices']['kits'][kit]['type'] == 'STATION':
                # print 'AlphaSense the sensor is'
                alphaDelta = dict()
                alphaDelta['CO'] = test['test']['devices']['kits'][kit]['alphasense']['CO']
                alphaDelta['NO2'] = test['test']['devices']['kits'][kit]['alphasense']['NO2']
                alphaDelta['O3'] = test['test']['devices']['kits'][kit]['alphasense']['O3']
                alphaDelta['SLOTS'] = test['test']['devices']['kits'][kit]['alphasense']['slots']
                readings[test_id]['devices'][kit]['alphasense'] = alphaDelta
                print ('\t**ALPHASENSE**')
                print ('\t' + str(alphaDelta))
                
            display(Markdown('Kit **{}** has been loaded'.format(kit)))
        ## Check if there's was a reference equipment during the test
        if test['reference']['available']:
            display(Markdown('### REFERENCE'))
            for reference in test['reference']['files']:
                display(Markdown('#### {}'.format(reference)))
                # print 'Reference during the test was'
                referenceDict =  dict()
                
                # Get the file name and frequency
                fileNameProc = test['reference']['files'][reference]['fileNameProc']
                frequency = test['reference']['files'][reference]['index']['frequency']
                
                # Check the index name
                timeIndex = test['reference']['files'][reference]['index']['name']
                location = test['reference']['files'][reference]['location']
                display(Markdown('Reference location **{}**'.format(location)))
                
                # Open it with pandas    
                fileData = join(testPath, fileNameProc)
                df = pd.read_csv(fileData, verbose=False, skiprows=[1]).set_index(timeIndex)
                df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
                df.sort_index(inplace=True)
                df = df.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
                df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
                
                ## Convert units
                # Get which pollutants are available in the reference
                pollutants = test['reference']['files'][reference]['channels']['pollutants']
                channels = test['reference']['files'][reference]['channels']['names']
                units = test['reference']['files'][reference]['channels']['units']
                
                for index in range(len(channels)):
                    
                    pollutant = pollutants[index]
                    channel = channels[index]
                    unit = units[index]
                    
                    # Get molecular weight and target units for the pollutant in question
                    for pollutantItem in pollutantLUT:
                        if pollutantItem[0] == pollutant:
                            molecularWeight = pollutantItem[1]
                            targetUnit = pollutantItem[2]
                            
                    convertionLUT = (['ppm', 'ppb', 1000],
                         ['mg/m3', 'ug/m3', 1000],
                         ['mg/m3', 'ppm', 24.45/molecularWeight],
                         ['ug/m3', 'ppb', 24.45/molecularWeight],
                         ['mg/m3', 'ppb', 1000*24.45/molecularWeight],
                         ['ug/m3', 'ppm', 1./1000*24.45/molecularWeight])
                    
                    # Get convertion factor
                    if unit == targetUnit:
                            convertionFactor = 1
                            print ('\tNo unit convertion needed for {}'.format(pollutant))
                    else:
                        for convertionItem in convertionLUT:
                            if convertionItem[0] == unit and convertionItem[1] == targetUnit:
                                convertionFactor = convertionItem[2]
                            elif convertionItem[1] == unit and convertionItem[0] == targetUnit:
                                convertionFactor = 1.0/convertionItem[2]
                        print ('\tConverting _{}_ from _{}_ to _{}_'.format(pollutant, unit, targetUnit))
                            
                    df.loc[:,pollutant + '_' + ref_append] = df.loc[:,channel]*convertionFactor
                    
                referenceDict['data'] = df
                readings[test_id]['devices'][reference] = referenceDict
                readings[test_id]['devices'][reference]['is_reference'] = True
                display(Markdown('**{}** reference has been loaded'.format(reference)))
        print ('------------------------------------------------------')
    return readings

def combine_data(list_of_datas, check_reference):
    dataframe_result = pd.DataFrame()
    for i in list_of_datas:
        dataframe = pd.DataFrame()
        dataframe = dataframe.combine_first(list_of_datas[i]['data'])
        # Check if there is reference
        if 'is_reference' in list_of_datas[i]:
            append = i
            prepend = 'REF_'
        else:
            append = i
            prepend = ''
        new_names = list()
        for name in dataframe.columns:
            # print name
            new_names.append(prepend + name + '_' + append)
        
        dataframe.columns = new_names
        dataframe_result = dataframe_result.combine_first(dataframe)
    return dataframe_result

# # Usage example
# for test in readings:
#     print test
#     for reading in readings[test]:
#         print reading
#         if 'alphasense' in readings[test]['devices'][reading]:
#             print 'The Kit with alphasense is {}'.format(reading)
#             display(readings[test][reading]['alphasense'])
#         if 'is_reference' in readings[test]['devices'][reading]:
#             print 'The reference is {}'.format(reading)
#             display(readings[test][reading]['data'].head(4))
            
# ## Usage Example
# slot = 2
# name = CHANNEL_NAME(currentSensorNames, 'GASES', slot, 'WORKING', 'BOARD_AUX')
# print name

# slot = 2
# name = CHANNEL_NAME(currentSensorNames, 'GASES', slot, 'AUXILIARY', 'BOARD_AUX')
# print name

# slot = 3
# name = CHANNEL_NAME(commitSensorNames, 'ALPHA', slot, 'WORKING', 'BOARD_AUX')
# print name

# slot = 2
# name = CHANNEL_NAME(commitSensorNames, 'ALPHA', slot, 'AUXILIARY', 'BOARD_AUX')
# print name

# name = CHANNEL_NAME(currentSensorNames, 'TEMPERATURE', 0, '?ONE', 'BOARD_AUX')
# print name

# name = CHANNEL_NAME(currentSensorNames, 'HUMIDITY', 0, '?ONE', 'BOARD_AUX')
# print name

# name = CHANNEL_NAME(commitSensorNames, 'TEMPERATURE', 0, '?ONE', 'BOARD_AUX')
# print name

# name = CHANNEL_NAME(commitSensorNames, 'HUMIDITY', 0, '?ONE', 'BOARD_AUX')
# print name

# typeSLOT = ('WORKING','AUXILIARY')

# namesNEW = list()
# for types in ('WORKING','AUXILIARY'):
#     for slot in (1,2,3):
#         namesNEW.append(CHANNEL_NAME(currentSensorNames, 'GASES', slot, types, 'BOARD_AUX'))
# namesNEW.append(CHANNEL_NAME(currentSensorNames, 'TEMPERATURE', 0, '?ONE', 'BOARD_AUX'))
# namesNEW.append(CHANNEL_NAME(currentSensorNames, 'HUMIDITY', 0, '?ONE', 'BOARD_AUX'))

# namesOLD = list()
# for types in ('WORKING','AUXILIARY'):
#     for slot in (1,2,3):
#         namesOLD.append(CHANNEL_NAME(commitSensorNames, 'ALPHA', slot, types, 'BOARD_AUX'))
# namesOLD.append(CHANNEL_NAME(commitSensorNames, 'TEMPERATURE', 0, '?ONE', 'BOARD_AUX'))
# namesOLD.append(CHANNEL_NAME(commitSensorNames, 'HUMIDITY', 0, '?ONE', 'BOARD_AUX'))