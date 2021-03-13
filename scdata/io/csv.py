from os import makedirs, listdir
from os.path import exists, join, dirname, realpath, splitext
from scdata.utils import std_out, localise_date, clean
from pandas import read_csv, to_datetime, to_numeric, option_context, DataFrame
import csv

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
    # df = df.apply(to_numeric, errors='coerce')
    df = df.astype(float, errors='ignore')

    # Resample
    df = df.resample(frequency).mean()

    # Remove na
    df = clean(df, clean_na, how = 'all')
    
    return df

def sdcard_concat(path, output = 'CONCAT.CSV', index_name = 'TIME', keep = True, ignore = 'CONCAT.CSV'):
    '''
        Loads files from local directory in text format, for instance
        SD card files with timestamp, sparse or concatenated
        Parameters
        ----------
            path: String
                Directory containing the folder
            output: String
                CONCAT.CSV
                Output name (csv file). If '' no output is saved, only
                returns a pandas.DataFrame()
            index_name: String
                'TIME'
                Name for the index of the pandas.DataFrame()
            keep: boolean
                True
                Keeps the header in the output file
            ignore: list
                CONCAT.CSV
                Ignores this file if present in the folder
        Returns
        -------
            Pandas dataframe
    '''

    concat = DataFrame()
    header_tokenized = dict()
    marked_for_revision = False

    for file in listdir(path):
        if file != output and file != ignore:
            std_out(f'Loading file: {file}')
            filename, _ = splitext(file)
            src_path = join(path, file)

            try:
                with open(src_path, 'r', newline = '\n') as csv_file:
                    header = csv_file.readlines()[0:4]
            except:
                ignore_file = True
                std_out(f'Ignoring file: {file}', 'WARNING')
                pass
            else:
                ignore_file = False

            if ignore_file: continue

            if keep:
                short_tokenized = header[0].strip('\r\n').split(',')
                unit_tokenized = header[1].strip('\r\n').split(',')
                long_tokenized = header[2].strip('\r\n').split(',')
                id_tokenized = header[3].strip('\r\n').split(',')

                for item in short_tokenized:
                    if item != '' and item not in header_tokenized.keys():
                        index = short_tokenized.index(item)
                        
                        header_tokenized[short_tokenized[index]] = dict()
                        header_tokenized[short_tokenized[index]]['unit'] = unit_tokenized[index]
                        header_tokenized[short_tokenized[index]]['long'] = long_tokenized[index]
                        header_tokenized[short_tokenized[index]]['id'] = id_tokenized[index]
            
            temp = read_csv(src_path, verbose=False, skiprows=range(1,4)).set_index("TIME")
            temp.index.rename(index_name, inplace=True)
            concat = concat.combine_first(temp)            

    columns = concat.columns

    ## Sort index
    concat.sort_index(inplace = True)

    ## Save it as CSV
    if output.endswith('.CSV') or output.endswith('.csv'):
        concat.to_csv(join(path, output))

        if keep:
            print ('Updating header')
            with open(join(path, output), 'r') as csv_file:
                content = csv_file.readlines()

                final_header = content[0].strip('\n').split(',')
                short_h = []
                units_h = []
                long_h = []
                id_h = []

                for item in final_header:
                    if item in header_tokenized.keys():
                        short_h.append(item)
                        units_h.append(header_tokenized[item]['unit'])
                        long_h.append(header_tokenized[item]['long'])
                        id_h.append(header_tokenized[item]['id'])

                content.pop(0)

                for index_content in range(len(content)):
                    content[index_content] = content[index_content].strip('\n')

                content.insert(0, ','.join(short_h))
                content.insert(1, ','.join(units_h))
                content.insert(2, ','.join(long_h))
                content.insert(3, ','.join(id_h))

            with open(join(path, output), 'w') as csv_file:
                print ('Saving file to:', output)
                wr = csv.writer(csv_file, delimiter = '\t')

                for row in content:
                    wr.writerow([row])

    return concat