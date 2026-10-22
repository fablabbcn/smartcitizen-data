''' Where tests are stored, and how they are found '''

import json
from datetime import datetime
from os import listdir, walk
from os.path import abspath, exists, isdir, join, sep
from re import escape, match

from pandas import DataFrame

from scdata._config import config
from scdata.tools.custom_logger import logger

# Tests are named <year>_<month>_<name>, from the date they were created
DATE_PREFIX = r'\d{4}_\d{2}_'

def full_name(name):
    ''' Adds the date prefix to the name, if it doesn't have it '''
    if match(DATE_PREFIX, name): return name
    current_date = datetime.now()
    return f'{current_date.year}_{str(current_date.month).zfill(2)}_{name}'

def default_root(name):
    ''' Folder in the cache for a test: config.paths['processed']/<year>/<month> '''
    year, month = full_name(name).split('_')[:2]
    return join(config.paths['processed'], year, month)

def in_cache(path):
    return abspath(path).startswith(abspath(config.paths['processed']) + sep)

def registry_path():
    ''' File with the folders of the tests stored out of config.paths['processed'] '''
    return join(config.paths['data'], 'tests.json')

def load_registry():
    ''' Registered folders that still have a test '''
    if not exists(registry_path()): return []
    try:
        with open(registry_path(), 'r') as file:
            registry = json.load(file)
    except ValueError:
        logger.warning(f'Invalid tests registry: {registry_path()}')
        return []
    if not isinstance(registry, list): return []
    return [path for path in registry if exists(join(path, 'test.json'))]

def register(path):
    ''' Keeps the test folder, so the test is found by name wherever it is stored '''
    path = abspath(path)
    if in_cache(path): return
    registry = load_registry()
    if path in registry: return
    registry.append(path)
    with open(registry_path(), 'w') as file:
        json.dump(registry, file, indent=4)

def read_meta(path):
    ''' Meta and number of devices in path/test.json, or None if it can't be read '''
    try:
        with open(join(path, 'test.json'), 'r') as file:
            tj = json.load(file)
        meta = dict(tj['meta'])
        meta['devices'] = len(tj.get('devices', []))
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning(f'Invalid test in {path}')
        return None
    return meta

def folders(root = None):
    '''
        Test folders: in config.paths['processed'], registered (stored in
        other folders) and directly in root, if given
    '''
    paths = []

    for root_dir, dirs, files in walk(config.paths['processed']):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        if 'test.json' in files:
            paths.append(abspath(root_dir))
            # The subfolders are the test data
            dirs[:] = []

    paths.extend(load_registry())

    # Not walked: root can be any folder
    if root is not None and isdir(root):
        for item in sorted(listdir(root)):
            path = abspath(join(root, item))
            if not item.startswith('.') and exists(join(path, 'test.json')):
                paths.append(path)

    # Without duplicates, in order
    return list(dict.fromkeys(paths))

def tests(pattern = None, root = None):
    '''
        Lists the stored tests

        Parameters
        ----------
        pattern: str
            None
            Only the tests whose name contains it (case insensitive)
        root: str
            None
            Folder with tests, to list them as well (registered tests are always listed)

        Returns
        ----------
        DataFrame with the name, type, number of devices and path of each test
    '''
    rows = []
    for path in folders(root):
        meta = read_meta(path)
        if meta is None: continue
        name = meta.get('name', '')
        if pattern is not None and pattern.lower() not in name.lower(): continue
        rows.append({'name': name, 'type': meta.get('type'),
                     'devices': meta['devices'], 'path': path})

    return DataFrame(rows, columns=['name', 'type', 'devices', 'path'])

def find(name, root = None):
    '''
        Folders of the tests named name, with or without the date prefix
    '''
    found = []
    for path in folders(root):
        meta = read_meta(path)
        if meta is None: continue
        stored = meta.get('name', '')
        if stored == name or (not match(DATE_PREFIX, name) and match(DATE_PREFIX + escape(name) + '$', stored)):
            found.append(path)
    return found
