import urllib2
import re
import os
from os.path import dirname, join
from os import getcwd
from sklearn.externals import joblib
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
    # Directory
    nameDictPath = join(getcwd(), 'sensorData/sensorNames.sav')

    try:
        # Read only 20000 chars
        data = urllib2.urlopen(_sensorsh).read(20000)
        # split it into lines
        data = data.split("\n") 
        sensorNames = dict()
        lineSensors = len(data)

        # Put everything in the dict
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

        # Save everything to the most recent one
        joblib.dump(sensorNames, nameDictPath)
    
    except:

        # Directory
        # Load sensors
        print 'No connection - Retrieving local version for sensors names'
        sensorNames = joblib.load(nameDictPath)

    return sensorNames

currentSensorNames = getSensorNames(currentSensorsh)
currentSensorNames[97] = dict()
currentSensorNames[97]['shortTitle'] = 'BATT_CHG_RATE'
currentSensorNames[97]['title'] = 'BATT_CHARGE_RATE'
currentSensorNames[97]['id'] = 97
currentSensorNames[97]['unit'] = 'mA'

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

            try:
                metadata = test['test']['devices']['kits'][kit]['metadata']
            except:
                metadata = ''
                pass
            targetSensorNames = list()
            testSensorNames = list()

            for item_test in metadata:
                id_test = metadata[item_test]['id']

                for item_target in currentSensorNames:
                    if currentSensorNames[item_target]['id'] == id_test and id_test != '0' and item_test not in testSensorNames:
                        targetSensorNames.append(currentSensorNames[item_target]['shortTitle'])
                        testSensorNames.append(item_test)            

            display(Markdown('#### {}'.format(kit)))
            # Get fileName
            fileNameProc = test['test']['devices']['kits'][kit]['fileNameProc']
            # frequency = test['test']['devices']['kits'][kit]['frequency']
            fileData = join(testPath, fileNameProc)
            location = test['test']['devices']['kits'][kit]['location']
            display(Markdown('Kit **{}** located **{}**'.format(kit, location)))
            
            # Create pandas dataframe
            df = pd.read_csv(fileData, verbose=False, skiprows=[1]).set_index('Time')
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
        
            df.sort_index(inplace=True)

            df = df.apply(pd.to_numeric,errors='coerce')            

            df.fillna(0)
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
                frequency_ref = test['reference']['files'][reference]['index']['frequency']
                if frequency != frequency_ref:
                    print 'resampling reference'

                # Check the index name
                timeIndex = test['reference']['files'][reference]['index']['name']
                location = test['reference']['files'][reference]['location']
                display(Markdown('Reference location **{}**'.format(location)))
                
                # Open it with pandas    
                fileData = join(testPath, fileNameProc)
                df = pd.read_csv(fileData, verbose=False).set_index(timeIndex)
                df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)
                df.sort_index(inplace=True)
                df = df.apply(pd.to_numeric,errors='coerce')            

                df.fillna(0)
                df = df.groupby(pd.Grouper(freq=frequency)).aggregate(np.mean)
                
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