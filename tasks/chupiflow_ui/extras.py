from os import environ, name
from os.path import expanduser, join, isdir

def get_dpath():

    # Check if windows
    _mswin = name == "nt"
    # Get user_home
    _user_home = expanduser("~")

    # Get .cache dir - maybe change it if found in config.json
    if _mswin:
        _ddir = environ["APPDATA"]
    elif 'XDG_CACHE_HOME' in environ:
        _ddir = environ['XDG_CACHE_HOME']
    else:
        _ddir = join(expanduser("~"), '.cache')

    dpath = join(_ddir, 'scdata', 'tasks')

    return dpath

def check_path(path):
    return isdir(path)