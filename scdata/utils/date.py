from pandas import to_datetime

def localise_date(date, location):
    """
    Localises a date if it's tzinfo is None, otherwise converts it to it.
    If the timestamp is tz-aware, converts it as well
    Parameters
    ----------
        date: string or datetime
            Date
        location: string
            Timezone string. i.e.: 'Europe/Madrid'
    Returns
    -------
        The date converted to 'UTC' and localised based on the timezone
    """    
    
    if date is not None:
        result_date = to_datetime(date, utc = True)
        if result_date.tzinfo is not None: 
            result_date = result_date.tz_convert(location)
        else:
            result_date = result_date.tz_localize(location)

    else: 
        result_date = None

    return result_date

def find_dates(dataframe):
    """
    Calculates minimum, maximum dates in the dataframe and the amount of days in between
    Parameters
    ----------
        dataframe: pd.DataFrame
            pandas dataframe with datetime index
    Returns
    -------
        Rounded down first day, rounded up last day and number of days between them
    """    

    range_days = (dataframe.index.max()-dataframe.index.min()).days
    min_date_df = dataframe.index.min().floor('D')
    max_date_df = dataframe.index.max().ceil('D')
    
    return min_date_df, max_date_df, range_days