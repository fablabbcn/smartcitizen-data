from os import makedirs, listdir
from os.path import exists, join, splitext
import csv

from scdata.tools.custom_logger import logger
from scdata.tools.date import localise_date
from scdata.tools.cleaning import clean
from pandas import read_csv, to_datetime, DataFrame
from scdata._config import config
from scdata.models import Metric

class CSVHandler:
    ''' Main implementation of the CSV data class '''

    def __init__(self, params):
        self.id = params.id
        self.params = params
        self.method = 'sync'
        self.data = DataFrame()
        self._metrics: List[Metric] = []
        self.latest_postprocessing = None
        if not self.__check__():
            raise FileExistsError(f'File not found: {self.params.path}')

    def __check__(self):
        return exists(self.params.path)

    @property
    def timezone(self):
        return self.params.timezone

    # This returns an empty list to avoid renaming CSVs
    @property
    def sensors(self):
        return []

    def update_latest_postprocessing(self, date):

        try:
            self.latest_postprocessing = date.to_pydatetime()
        except:
            return False
        else:
            logger.info(f"Updated latest_postprocessing to: {self.latest_postprocessing}")
            return True

        logger.info('Nothing to update')

        return True

    def get_data(self, **kwargs):
        self.data = read_csv_file(self.params.path,
            timezone= self.timezone,
            frequency= kwargs['frequency'],
            clean_na= kwargs['clean_na'],
            index_name= self.params.index,
            skiprows=self.params.header_skip,
            sep=self.params.separator,
            tzaware=self.params.tzaware,
            resample=kwargs['resample']
        )

        return self.data

def export_csv_file(path, file_name, df, forced_overwrite=False):
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
        logger.info('File saved to: \n' + path + '/' + str(file_name) + '.csv')
    else:
        logger.error("File Already exists - delete it first, I was not asked to overwrite anything!")
        return False
    return True

def read_csv_file(path, timezone, frequency=None, clean_na=None, index_name='', skiprows=None, sep=',', encoding='utf-8', tzaware=True, resample=True):
    """
    Reads a csv file and adds cleaning, localisation and resampling and puts it into a pandas dataframe
    Parameters
    ----------
        path: String
            File path for csv file
        timezone: String
            Time zone for the csv file
        frequency: String
            None
            Frequency in pandas format of the desired output
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

    df = read_csv(path, skiprows=skiprows, sep=sep,
                encoding=encoding, encoding_errors='ignore')

    flag_found = False
    if type(index_name) == str:
        # Single joint index
        for column in df.columns:
            if index_name in column:
                df = df.set_index(column)
                flag_found = True
                break
    elif type(index_name) == list:
        # Composite index (for instance, DATE and TIME in different columns)
        for iname in index_name:
            if iname not in df.columns:
                logger.error(f'{iname} not found in columns')
                return None
        joint_index_name = '_'.join(index_name)
        df[joint_index_name] = df[index_name].agg(' '.join, axis=1)
        df = df.set_index(joint_index_name)
        df.drop(index_name, axis=1, inplace=True)
        flag_found = True

    if not flag_found:
        logger.error('Index not found. Cannot reindex')
        return None

    # Set index
    df.index = localise_date(df.index, timezone, tzaware=tzaware)
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
    if (resample):
        logger.info ('Resampling', 'INFO')
        df = df.resample(frequency).mean()

    # Remove na
    df = clean(df, clean_na, how = 'all')

    return df

def sdcard_concat(path, output = 'CONCAT.CSV', index_name = 'TIME', keep = True, ignore = ['CONCAT.CSV', 'INFO.TXT'], **kwargs):
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
        if file != output and file not in ignore:
            logger.info(f'Loading file: {file}')
            filename, _ = splitext(file)
            src_path = join(path, file)

            try:
                with open(src_path, 'r', newline = '\n', errors = 'replace') as csv_file:
                    header = csv_file.readlines()[0:4]
            except:
                ignore_file = True
                logger.warning(f'Ignoring file: {file}')
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

            temp = read_csv(src_path, verbose=False, skiprows=range(1,4),
                            encoding_errors='ignore', na_values=config._ignore_na_values).set_index("TIME")
            temp = clean(temp, clean_na='drop', how='all')
            temp.index.rename(index_name, inplace=True)
            concat = concat.combine_first(temp)

    columns = concat.columns

    ## Sort index
    concat.sort_index(inplace = True)

    # Rename case
    if 'rename_to_blueprint' in kwargs:
        rename = kwargs['rename_to_blueprint']
    else:
        rename = False

    if 'blueprint' in kwargs:
        rename_bp = kwargs['blueprint']
        if rename_bp not in config.blueprints:
            logger.warning('Blueprint not in config. Cannot rename')
            rename = False
    else:
        logger.info('No blueprint specified')
        rename = False

    if rename:
        logger.warning('Keep in mind that renaming doesnt change the units')
        rename_d = dict()
        for old_key in header_tokenized:
            for key, value in config.blueprints[rename_bp]['sensors'].items():
                if value['id'] == header_tokenized[old_key]['id'] and old_key != key:
                    rename_d[old_key] = key
                    break

        for old_key in rename_d:
            logger.info(f'Renaming {old_key} to {rename_d[old_key]}')
            header_tokenized[rename_d[old_key]] = header_tokenized.pop(old_key)
            concat.rename(columns=rename_d, inplace=True)

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
