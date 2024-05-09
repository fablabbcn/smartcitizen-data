from requests import get
from pandas import json_normalize
from scdata.tools.custom_logger import logger
from scdata._config import config

def get_elevation(_lat = None, _long = None):
    '''
        script for returning elevation from lat, long,
        based on open elevation data
        which in turn is based on SRTM - elevation in m
        From:
        https://stackoverflow.com/questions/19513212/can-i-get-the-altitude-with-geopy-in-python-with-longitude-latitude
    '''
    if _lat is None or _long is None: return None

    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={_lat},{_long}')

    # Request with a timeout for slow responses
    error = False
    try:
        r = get(query, timeout = config._timeout)
    except:
        logger.info(f'Cannot get altitude from {query}')
        error = True
        pass

    if error: return None

    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else:
        elevation = None

    try:
        elevation = int(elevation)
    except:
        elevation = None
        pass

    return elevation
