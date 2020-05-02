from os.path import join
from urllib.request import urlopen
from scdata.utils import std_out
from scdata._config import config

import json
from re import sub
from traceback import print_exc

def get_firmware_names(sensorsh, json_path, file_name):
    # Directory
    names_dict = join(json_path, file_name + '.json')
    
    if config.reload_firmware_names:
        try:
            # Read only 20000 chars
            data = urlopen(sensorsh).read(20000).decode('utf-8')
            # split it into lines
            data = data.split('\n')
            sensor_names = dict()
            line_sensors = len(data)
            for line in data:
                if 'class AllSensors' in line:
                    line_sensors = data.index(line)
                    
                if data.index(line) > line_sensors:

                    if 'OneSensor' in line and '{' in line and '}' in line and '/*' not in line:
                        # Split commas
                        line_tokenized =  line.strip('').split(',')

                        # Elimminate unnecessary elements
                        line_tokenized_sublist = list()
                        for item in line_tokenized:
                                item = sub('\t', '', item)
                                item = sub('OneSensor', '', item)
                                item = sub('{', '', item)
                                item = sub('}', '', item)
                                #item = sub(' ', '', item)
                                item = sub('"', '', item)
                                
                                if item != '' and item != ' ':
                                    while item[0] == ' ' and len(item)>0: item = item[1:]
                                line_tokenized_sublist.append(item)
                        line_tokenized_sublist = line_tokenized_sublist[:-1]

                        if len(line_tokenized_sublist) > 2:
                                shortTitle = sub(' ', '', line_tokenized_sublist[3])
                                if len(line_tokenized_sublist)>9:
                                    sensor_names[shortTitle] = dict()
                                    sensor_names[shortTitle]['SensorLocation'] = sub(' ', '', line_tokenized_sublist[0])
                                    sensor_names[shortTitle]['id'] = sub(' ','', line_tokenized_sublist[5])
                                    sensor_names[shortTitle]['title'] = line_tokenized_sublist[4]
                                    sensor_names[shortTitle]['unit'] = line_tokenized_sublist[-1]
            # Save everything to the most recent one
            with open(names_dict, 'w') as fp:
                json.dump(sensor_names, fp)
            std_out('Saved updated sensor names and dumped into ' + names_dict, 'SUCCESS')

        except:
            # Load sensors
            print_exc()
            with open(names_dict) as handle:
                sensor_names = json.loads(handle.read())
            std_out('Error. Retrieving local version for sensors names', 'WARNING')

    else:
        std_out('Retrieving local version for sensors names')
        with open(names_dict) as handle:
            sensor_names = json.loads(handle.read())
        if sensor_names is not None: std_out('Local version of sensor names loaded', 'SUCCESS')

    return sensor_names