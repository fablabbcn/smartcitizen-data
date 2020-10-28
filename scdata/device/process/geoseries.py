from geopy.distance import distance
from math import isnan

def is_within_circle(x, within, _lat_name = 'GPS_LAT', _long_name = 'GPS_LONG'):
    ''' 
    Returns whether or not a line in pd.DataFrame() is geolocated within a circle
    Parameters
    ----------
        within: tuple
            Empty tuple
            Gets the devices within a circle center on lat, long with a radius_meters
            within = tuple(lat, long, radius_meters)
        _lat_name: str, optional 
            GPS_LAT
            Column name for latitude in dataframe
        _long_name: str, optional
            GPS_LONG
            Column name for long in dataframe
    Returns
    -------
        pandas series containing bool defining wether or not each dataframe[:, [lat_name, long_name]] are within circle of basepoint(lat, long)
    '''

    if isnan(x[_lat_name]): return False
    if isnan(x[_long_name]): return False

    return distance((within[0], within[1]), (x[_lat_name], x[_long_name])).m < within[2]