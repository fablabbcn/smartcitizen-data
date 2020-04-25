from pandas import read_csv, to_datetime, to_numeric
from scdata.utils.out import std_out

def read_csv_file(file_path, location, frequency, clean_na = None, index_name = '', skiprows = None, sep = ',', encoding = 'utf-8'):
    '''
        Reads a csv file and adds cleaning, localisation and resampling and puts it into a pandas dataframe
        Parameters
        ----------
            file_path: String
                File path for csv file
            location: String
                Time zone for the csv file
            clean_na: String or None
                None
                Whether to perform clean_na or not. Either None, 'fill' or 'drop'
            index_name: String
                ''
                Name of the column to set an index in the dataframe
            skiprows: list or None
                None
                List of rows to skip (same as skiprows in pandas.read_csv)
            sep: String
                ','
                Separator (same as sep in pandas.read_csv)
            encoding: String
                'utf-8'
                Encoding of the csv file
        Returns
        -------
            Pandas dataframe
    '''    

    # Read pandas dataframe
    df = read_csv(file_path, verbose = False, skiprows = skiprows, sep = ',', encoding = encoding)

    for column in df.columns:
        if index_name in column: 
            df = df.set_index(column)
            break

    # Set index
    df.index = to_datetime(df.index).tz_localize('UTC').tz_convert(location)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort index
    df.sort_index(inplace=True)
    
    # Drop unnecessary columns
    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
    
    # Check for weird things in the data
    df = df.apply(to_numeric,errors='coerce')   
    
    # Resample
    df = df.resample(frequency, limit = 1).mean()

    # Remove na
    if clean_na is not None:
        if clean_na == 'fill':
            df = df.fillna(method='ffill')
        elif clean_na == 'drop':
            df.dropna(axis = 0, how='all', inplace=True)

    return df