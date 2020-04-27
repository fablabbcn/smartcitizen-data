from os import makedirs
from os.path import exists
from scdata.utils.out import std_out

def export_csv_file(path, file_name, df, forced_overwrite = False):
    '''
        Exports pandas dataframe to a csv file
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