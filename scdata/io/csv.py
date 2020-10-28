from os import makedirs
from os.path import exists
from scdata.utils import std_out, localise_date, clean
from pandas import read_csv, to_datetime, to_numeric, option_context

def export_csv_file(path, file_name, df, forced_overwrite = False):
    '''
    Exports pandas dataframe to a csv file
    Parameters
    ----------
        path: String
            Directory path
        file_name: String
            File name for the resulting csv
        df: pandas.DataFrame
            Dataframe to export
        forced_overwrite: boolean
            False
            If file exists, overwrite it or not
    Returns
    ---------
        True if exported, False if not (if file exists returns False)
    '''

    # If path does not exist, create it
    if not exists(path):
        makedirs(path)

    # If file does not exist 
    if not exists(path + '/' + str(file_name) + '.csv') or forced_overwrite:
        df.to_csv(path + '/' + str(file_name) + '.csv', sep=",")
        std_out('File saved to: \n' + path + '/' + str(file_name) +  '.csv', 'SUCCESS')
    else:
        std_out("File Already exists - delete it first, I was not asked to overwrite anything!", 'ERROR')
        return False
    
    return True

def read_csv_file(file_path, location, frequency, clean_na = None, index_name = '', skiprows = None, sep = ',', encoding = 'utf-8'):
    """
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
    """  

    # Read pandas dataframe

    df = read_csv(file_path, verbose = False, skiprows = skiprows, sep = ',', encoding = encoding)

    flag_found = False
    for column in df.columns:
        if index_name in column: 
            df = df.set_index(column)
            flag_found = True
            break

    if not flag_found:
        std_out('Index not found. Cannot reindex', 'ERROR')
        return None

    # Set index
    df.index = localise_date(df.index, location)
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort index
    df.sort_index(inplace=True)
    
    # Drop unnecessary columns
    df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
    
    # Check for weird things in the data
    df = df.apply(to_numeric, errors='coerce')   
    
    # Resample
    df = df.resample(frequency).mean()

    # Remove na
    df = clean(df, clean_na, how = 'all')
    
    return df    