from os import walk
from os.path import join
import yaml
from scdata._config import config

def get_tests_log(deep_description = False):
    '''
        Gets the tests in the given dir, looking for test_description.yaml
    '''

    # Get available tests in the data folder structure
    tests = dict()

    for root, dirs, files in walk(config.paths['processed']):
        for file in files:
            if file.endswith(".yaml"):
                test_name = root.split('/')[-1]
                if test_name.startswith('.'): continue
                
                tests[test_name] = dict()
                tests[test_name]['path'] = root
                
                if deep_description == True:
                    filePath = join(root, file)
                    with open(filePath, 'r') as stream:
                        yamlFile = yaml.load(stream, Loader = yaml.FullLoader)
                        for key in yamlFile.keys():
                            if key == 'devices': continue
                            tests[test_name][key] = yamlFile[key]

    return tests