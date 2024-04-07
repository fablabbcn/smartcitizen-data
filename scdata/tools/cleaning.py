def clean(df, clean_na = None, how = 'all'):
    """
    Helper function for cleaning nan in a pandas.DataFrame
    Parameters
    ----------
        df: pandas.DataFrame
            The dataframe to clean
        clean_na: None or string
            type of nan cleaning. If not None, can be 'drop' or 'fill'
        how: 'string'
            Same as how in dropna, fillna. Can be 'any', or 'all'
    Returns
    -------
        Clean dataframe
    """

    if clean_na is not None:
        if clean_na == 'drop':
            df.dropna(axis = 0, how = how, inplace = True)
        elif clean_na == 'fill':
            df = df.fillna(method = 'bfill').fillna(method = 'ffill')
    return df
