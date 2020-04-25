from pandas import to_datetime

def localise_date(date, location):
    """
    Localises a date, converted to 'UTC'
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