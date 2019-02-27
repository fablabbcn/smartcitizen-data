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
            alphasense['slots'] = test['test']['devices']['kits'][kit]['alphasense']['slots']
            _readings[test_id]['devices'][kit]['alphasense'] = alphasense
            print ('\t\t**ALPHASENSE**')
            print ('\t\t' + str(alphasense))
            
        display(Markdown('Kit **{}** has been loaded'.format(kit)))
    
    ## Check if there's was a reference equipment during the test
    print (test['test']['devices'].keys())
    if 'reference' in test['test']['devices'].keys():
            refAvail = True
    else:
        refAvail = False

    if refAvail:
        display(Markdown('### REFERENCE'))
        for reference in test['test']['devices']['reference']:
            display(Markdown('#### {}'.format(reference)))
            # print 'Reference during the test was'
            referenceDict =  dict()
            
            # Get the file name and frequency
            fileNameProc = test['test']['devices']['reference'][reference]['fileNameProc']
            frequency_ref = test['test']['devices']['reference'][reference]['index']['frequency']
            if target_raster != frequency_ref:
                print ('Resampling reference')

            # Check the index name
            timeIndex = test['test']['devices']['reference'][reference]['index']['name']
            location = test['test']['devices']['reference'][reference]['location']
            display(Markdown('Reference location **{}**'.format(location)))
            
            # Open it with pandas    
            filePath = join(testPath, fileNameProc)
            df = readDataframeCsv(filePath, location, target_raster, clean_na, clean_na_method, [], [], timeIndex)
            
            ## Convert units
            # Get which pollutants are available in the reference
            pollutants = test['test']['devices']['reference'][reference]['channels']['pollutants']
            channels = test['test']['devices']['reference'][reference]['channels']['names']
            units = test['test']['devices']['reference'][reference]['channels']['units']
            
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

def combine_data(list_of_datas, check_reference, ignore_keys = []):
    dataframe_result = pd.DataFrame()

    for i in list_of_datas:
        if i not in ignore_keys:

            dataframe = pd.DataFrame()
            dataframe = dataframe.combine_first(list_of_datas[i]['data'])

            append = i
            prepend = ''
            new_names = list()
            for name in dataframe.columns:
                # print name
                new_names.append(prepend + name + '_' + append)
            
            dataframe.columns = new_names
            dataframe_result = dataframe_result.combine_first(dataframe)

    return dataframe_result

class test_object:
    
    def __init__(self, ID):
        self.ID = ID
        self.yaml = {}
        self.yaml['test'] = dict()
        self.yaml['test']['id'] = ID
        self.yaml['test']['devices'] = dict()
        self.yaml['test']['devices']['kits'] = dict()       
    
    def add_details(self, project = 'smartcitizen', commit = '', author = '', type_test = '', report = '', comment = ''):
        try:
            self.yaml['test']['project'] = project
            self.yaml['test']['commit'] = commit
            self.yaml['test']['author'] = author
            self.yaml['test']['type_test'] = type_test
            self.yaml['test']['report'] = report
            self.yaml['test']['comment'] = markdown.markdown(comment)
            print ('Add details OK')
        except:
            print ('Add device NOK')
            pass

    def add_device(self, device, device_type = 'KIT', sck_version = '2.0', pm_sensor = '', alphasense = {}, device_history = None,location = 'Europe/Madrid'):
        try:
            self.yaml['test']['devices']['kits'][device] = dict()
            self.yaml['test']['devices']['kits'][device]['type'] = device_type
            self.yaml['test']['devices']['kits'][device]['SCK'] = sck_version
            self.yaml['test']['devices']['kits'][device]['PM'] = pm_sensor
            self.yaml['test']['devices']['kits'][device]['location'] = location
            #### Alphasense
            if alphasense != {}:
                self.yaml['test']['devices']['kits'][device]['alphasense'] = alphasense
            elif device_history != None:
                self.yaml['test']['devices']['kits'][device]['alphasense'] = sensorsData[device_history]['gas_pro_board']
                
            print ('Add device {} OK'.format(device))
        except:
            print ('Add device {} NOK'.format(device))
            pass

    def device_files(self, device, fileNameRaw = '', fileNameInfo = '', frequency = '1Min', type_file = 'csv_new'):
        try:
            self.yaml['test']['devices']['kits'][device]['fileNameRaw'] = fileNameRaw
            self.yaml['test']['devices']['kits'][device]['fileNameInfo'] = fileNameInfo
            fileNameProc = (self.yaml['test']['id'] + '_' + self.yaml['test']['devices']['kits'][device]['type'] + '_' + str(device) + '.csv')
            self.yaml['test']['devices']['kits'][device]['fileNameProc'] = fileNameProc
            self.yaml['test']['devices']['kits'][device]['frequency'] = frequency
            self.yaml['test']['devices']['kits'][device]['type_file'] = type_file  
            print ('Add device files {} OK'.format(device))
        
        except:
            print ('Add device files {} NOK'.format(device))
            pass
    
    def add_reference(self, reference, fileNameRaw = '', index = {}, channels = {}, location = ''):
        print ('Adding reference: {}'.format(reference))
        if 'reference' not in self.yaml['test']['devices']:
            self.yaml['test']['devices']['reference'] = dict()
        
        self.yaml['test']['devices']['reference'][reference] = dict()
        self.yaml['test']['devices']['reference'][reference]['fileNameRaw'] = fileNameRaw
        self.yaml['test']['devices']['reference'][reference]['fileNameProc'] = self.yaml['test']['id'] + '_' + str(reference) + '_REF.csv'
        self.yaml['test']['devices']['reference'][reference]['index'] = index
        self.yaml['test']['devices']['reference'][reference]['channels'] = channels
        self.yaml['test']['devices']['reference'][reference]['location'] = location
    
    def process_files(self, _rootDirectory, _newpath):
        
        def get_raw_files():
                list_raw_files = []
                
                if 'kits' in self.yaml['test']['devices']:
                    for kit in self.yaml['test']['devices']['kits']:
                        list_raw_files.append(self.yaml['test']['devices']['kits'][kit]['fileNameRaw'])
                        
                if 'references' in self.yaml['test']['devices']:
                    for reference in self.yaml['test']['devices']['reference']:
                        list_raw_files.append(self.yaml['test']['devices']['references'][reference]['fileNameRaw'])
                        
                return list_raw_files    
        
        def copy_raw_files(_raw_src_path, _raw_dst_path, _list_raw_files):
            
                try: 

                    for item in _list_raw_files:
                        s = join(_raw_src_path, item)
                        d = join(_raw_dst_path, item)
                        copyfile(s, d)
                    
                    return True
                
                except:

                    return False
                
        def date_parser(s, a):
            return parser.parse(s).replace(microsecond=int(a[-3:])*1000)
    
        # Define Paths
        raw_src_path = join(_rootDirectory, 'data', 'raw')
        raw_dst_path = join(_newpath, 'RAW_DATA')    
        
        # Create Paths
        if not os.path.exists(raw_dst_path):
            os.makedirs(raw_dst_path)
        
        list_raw_files = get_raw_files()
        # Copy raw files and process data
        if copy_raw_files(raw_src_path, raw_dst_path, list_raw_files):
            # Process references
            if 'reference' in self.yaml['test']['devices']:
                for reference in self.yaml['test']['devices']['reference']:
                    print ('Processing reference: {}'.format(reference))
                    src_path = join(raw_src_path, self.yaml['test']['devices']['reference'][reference]['fileNameRaw'])
                    dst_path = join(_newpath, self.yaml['test']['id'] + '_' + str(reference) + '_REF.csv')
                    
                    # Time Name
                    timeName = self.yaml['test']['devices']['reference'][reference]['index']['name']
                    
                    # Load Dataframe
                    df = pd.read_csv(src_path, verbose=False, skiprows=[1]).set_index(timeName)
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    
                    df = df.groupby(pd.Grouper(freq = self.yaml['test']['devices']['reference'][reference]['index']['frequency'])).aggregate(np.mean)
                    
                    # Remove Duplicates and drop unnamed columns
                    df = df[~df.index.duplicated(keep='first')]
                    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
                    
                    # Export to csv in destination path
                    df.to_csv(dst_path, sep=",")
                    
            
            # Process kits
            if 'kits' in self.yaml['test']['devices']:
                for kit in self.yaml['test']['devices']['kits']:
                    print ('Processing device: {}'.format(kit))
                    src_path = join(raw_src_path, self.yaml['test']['devices']['kits'][kit]['fileNameRaw'])
                    dst_path = join(_newpath, self.yaml['test']['id'] + '_' + self.yaml['test']['devices']['kits'][kit]['type'] + '_' + str(kit) + '.csv')
                    
                    # Read file csv
                    if self.yaml['test']['devices']['kits'][kit]['type_file'] == 'csv_new':
                        skiprows_pd = range(1, 4)
                        index_name = 'TIME'
                        df = pd.read_csv(src_path, verbose=False, skiprows=skiprows_pd, encoding = 'utf-8', sep=',')

                    elif self.yaml['test']['devices']['kits'][kit]['type_file'] == 'csv_old':
                        index_name = 'Time'
                        df = pd.read_csv(src_path, verbose=False, encoding = 'utf-8')
                        
                    elif self.yaml['test']['devices']['kits'][kit]['type_file'] == 'csv_ms':
                        index_name = 'Time'
                        df = pd.read_csv(src_path, verbose=False, encoding = 'utf-8', parse_dates=[[0,1]], date_parser=date_parser)
                    
                    # Find name in case of extra weird characters
                    for column in df.columns:
                        if index_name in column: index_found = column
                            
                    df.set_index(index_found, inplace = True)
                    df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(self.yaml['test']['devices']['kits'][kit]['location'])
                    df.sort_index(inplace=True)
                            
                    # Remove Duplicates and drop unnamed columns
                    df = df[~df.index.duplicated(keep='first')]
                    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
                        
                    df.to_csv(dst_path, sep=",")
                    
                    ## Import units and ids
                    if self.yaml['test']['devices']['kits'][kit]['type_file'] == 'csv_new':
                        dict_header = dict()
                        with open(src_path, 'rb') as csvfile:
                            readercsv = csv.reader(csvfile, delimiter = ',')
                            line = 0
                        
                            header = next(readercsv)[1:]
                            unit = next(readercsv)[1:]
                            ids = next(readercsv)[1:]
                        
                            for key in header:
                                dict_header[key] = dict()
                                dict_header[key]['unit'] = unit[header.index(key)]
                                dict_header[key]['id'] = ids[header.index(key)]
                            
                            self.yaml['test']['devices']['kits'][kit]['metadata'] = dict_header
                    
                    ## Load txt info
                    if self.yaml['test']['devices']['kits'][kit]['fileNameInfo'] != '':
                        src_path_info = join(raw_src_path, self.yaml['test']['devices']['kits'][kit]['fileNameInfo'])
                        dict_info = dict()
                        with open(src_path_info, 'rb') as infofile:
                            for line in infofile:
                                line = line.strip('\r\n')
                                splitter = line.find(':')
                                dict_info[line[:splitter]]= line[splitter+2:] # Accounting for the space
                           
                        self.yaml['test']['devices']['kits'][kit]['info'] = dict_info
                
            
            # Create yaml with test description
            with open(join(_newpath, 'test_description.yaml'), 'w') as yaml_file:
                yaml.dump(self.yaml, yaml_file)
                
            print ('Test Creation Finished')