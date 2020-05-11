''' Implementation of csv export for devices in test '''

from os.path import join
from scdata.utils import std_out

def to_csv(self, path = None, forced_overwrite = False):
    """
    Exports devices in test to desired path
    Parameters
    ----------
        path: string
        	None
            The path (directory) to export the csv(s) into. If None, exports to test_path/processed/
        forced_overwrite: boolean
        	False
            To overwrite existing files
    Returns
    -------
        True if export successul
    """	
    export_ok = True

    if path is None: epath = join(self.path, 'processed')
    else: epath = path

    # Export to csv
    for device in self.devices.keys():
        export_ok &= self.devices[device].export(epath, forced_overwrite = forced_overwrite)

    if export_ok: std_out(f'Test {self.full_name} exported successfully', 'SUCCESS')
    else: std_out(f'Test {self.full_name} not exported successfully', 'ERROR')

    return export_ok

def desc_to_html(self, path = None):
    '''
    Generates an html description for the test
    Parameters
    ----------
        path:
            Directory to export it to
    Returns
    ----------
        True if successful export
    '''
    print ('Not yet')

    return False    