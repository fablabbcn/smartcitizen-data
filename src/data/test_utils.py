from urllib.request import urlopen
import re
import os
from os.path import join
from os import getcwd
from sklearn.externals import joblib
import yaml
from IPython.display import display, Markdown
from tzwhere import tzwhere
import pandas as pd
import numpy as np

pollutantLUT = (['CO', 28, 'ppm'],
                ['NO', 30, 'ppb'],
                ['NO2', 46, 'ppb'],
                ['O3', 48, 'ppb'])

ref_append = 'REF'

def getSensorNames(_sensorsh, nameDictPath):
    try:
        # Directory
        nameDictPath = join(nameDictPath, 'sensorNames.sav')

        # Read only 20000 chars
        data = urlopen(_sensorsh).read(20000).decode('utf-8')
        # split it into lines
        data = data.split('\n')
        sensorNames = dict()
        lineSensors = len(data)
        for line in data:
            if 'class AllSensors' in line:
                lineSensors = data.index(line)
                
            if data.index(line) > lineSensors:
                
                if 'OneSensor' in line and '{' in line and '}' in line and '/*' not in line:
                    # Split commas
                    lineTokenized =  line.strip('').split(',')

                    # Elimminate unnecessary elements
                    lineTokenizedSub = list()
                    for item in lineTokenized:
                            item = re.sub('\t', '', item)
                            item = re.sub('OneSensor', '', item)
                            item = re.sub('{', '', item)
                            item = re.sub('}', '', item)
                            #item = re.sub(' ', '', item)
                            item = re.sub('"', '', item)
                            
                            if item != '' and item != ' ':
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
                            # print (sensorNames[sensorID])
        
        # Save everything to the most recent one
        joblib.dump(sensorNames, nameDictPath)
        print ('Loaded updated sensor names and dumped into', nameDictPath)
    
    except:

        # Directory
        # Load sensors
        print ('No connection - Retrieving local version for sensors names')
        sensorNames = joblib.load(nameDictPath)

    return sensorNames

def getTests(directory):
    # Get available tests in the data folder structure
    tests = dict()
    mydir = join(directory, 'processed')
    for root, dirs, files in os.walk(mydir):
        for _file in files:
            if _file.endswith(".yaml"):
                filePath = join(root, _file)
                stream = open(filePath)
                yamlFile = yaml.load(stream)
                tests[yamlFile['test']['id']] = root
    return tests

def readDataframeCsv(filePath, location, target_raster, clean_na, clean_na_method, targetNames, testNames, refIndex = 'Time'):

        # Create pandas dataframe
        df = pd.read_csv(filePath, verbose=False, skiprows=[1])
        if 'Time' in df.columns:
            df = df.set_index('Time')
        elif 'TIME' in df.columns:
            df = df.set_index('TIME')
        else:
            refIndex
            print ('No known index found')

        # Set index
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(location)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Sort index
        df.sort_index(inplace=True)

        # Drop unnecessary columns
        df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
        
        # Check for weird things in the data
        df = df.apply(pd.to_numeric,errors='coerce')   
        
        # Resample
        df = df.resample(target_raster).mean()
        
        # Remove na
        if clean_na:
            if clean_na_method == 'fill':
                df = df.fillna(method='bfill').fillna(method='ffill')
            elif clean_na_method == 'drop':
                df = df.dropna()
        
        # Create dictionary and add it to the _readings key
        if len(targetNames) == len(testNames) and len(targetNames) > 0:
            for i in range(len(targetNames)):
                if not (testNames[i] == '') and not (testNames[i] == targetNames[i]) and testNames[i] in df.columns:
                    df.rename(columns={testNames[i]: targetNames[i]}, inplace=True)
                    display(Markdown('Renaming column *{}* to *{}*'.format(testNames[i], targetNames[i])))

        return df

def loadTest(testPath, target_raster, currentSensorNames, clean_na = True, clean_na_method = 'fill'):

    _readings = dict()

    # Find Yaml
    filePath = join(testPath, 'test_description.yaml')
    with open(filePath, 'r') as stream:
        test = yaml.load(stream)
    
    test_id = test['test']['id']
        
    _readings[test_id] = dict()
    _readings[test_id]['devices'] = dict()
    
    print('------------------------------------------------------')
    display(Markdown('## Test Load'))

    display(Markdown('Loading test **{}**'.format(test_id)))
    # display(Markdown('Test performed with commit **{}**'.format(_readings[test_id]['commit_hash'])))
    display(Markdown(test['test']['comment']))
    display(Markdown('### KIT'))

    # Open all kits
    for kit in test['test']['devices']['kits']:

        try:
            metadata = test['test']['devices']['kits'][kit]['metadata']
        except:
            metadata = ''
            pass
        
        # List for names conversion
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
        
        filePath = join(testPath, fileNameProc)
        location = test['test']['devices']['kits'][kit]['location']
        display(Markdown('Kit **{}** located **{}**'.format(kit, location)))
        
        # Create dict for kit
        kitDict = dict()
        kitDict['location'] = location
        kitDict['data'] = readDataframeCsv(filePath, location, target_raster, clean_na, clean_na_method, targetSensorNames, testSensorNames)
        
        _readings[test_id]['devices'][kit] = kitDict
        
        ## Check if it's a STATION and retrieve alphadelta codes
        if test['test']['devices']['kits'][kit]['type'] == 'STATION':
            # print 'AlphaSense the sensor is'
            alphasense = dict()
            alphasense['CO'] = test['test']['devices']['kits'][kit]['alphasense']['CO']
            alphasense['NO2'] = test['test']['devices']['kits'][kit]['alphasense']['NO2']
            alphasense['O3'] = test['test']['devices']['kits'][kit]['alphasense']['O3']
            alphasense['SLOTS'] = test['test']['devices']['kits'][kit]['alphasense']['slots']
            _readings[test_id]['devices'][kit]['alphasense'] = alphasense
            print ('\t\t**ALPHASENSE**')
            print ('\t\t' + str(alphasense))
            
        display(Markdown('Kit **{}** has been loaded'.format(kit)))
    
    ## Check if there's was a reference equipment during the test
    if 'reference' in test.keys():
        if 'available' in test['reference'].keys():
            if test['reference']['available'] == True:
                refAvail = True
            else:
                refAvail = False
        else:
            refAvail = False
    else:
        refAvail = False

    if refAvail:
        display(Markdown('### REFERENCE'))
        for reference in test['reference']['files']:
            display(Markdown('#### {}'.format(reference)))
            # print 'Reference during the test was'
            referenceDict =  dict()
            
            # Get the file name and frequency
            fileNameProc = test['reference']['files'][reference]['fileNameProc']
            frequency_ref = test['reference']['files'][reference]['index']['frequency']
            if target_raster != frequency_ref:
                print ('Resampling reference')

            # Check the index name
            timeIndex = test['reference']['files'][reference]['index']['name']
            location = test['reference']['files'][reference]['location']
            display(Markdown('Reference location **{}**'.format(location)))
            
            # Open it with pandas    
            filePath = join(testPath, fileNameProc)
            df = readDataframeCsv(filePath, location, target_raster, clean_na, clean_na_method, [], [], timeIndex)
            
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
                    display(Markdown('Converting *{}* from *{}* to *{}*'.format(pollutant, unit, targetUnit)))
                
                df.loc[:,pollutant + '_' + ref_append] = df.loc[:,channel]*convertionFactor
                
            referenceDict['data'] = df
            _readings[test_id]['devices'][reference] = referenceDict
            _readings[test_id]['devices'][reference]['is_reference'] = True
            display(Markdown('**{}** reference has been loaded'.format(reference)))
        
        print ('------------------------------------------------------')
    return _readings

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