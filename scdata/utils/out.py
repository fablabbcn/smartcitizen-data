from termcolor import colored
from scdata._config import config
from datetime import datetime

def std_out(msg, mtype = None, force = False):
    out_level = config._out_level
    if config._timestamp == True:
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        stamp = ''
    # Output levels:
    # 'QUIET': nothing, 
    # 'NORMAL': warn, err
    # 'DEBUG': info, warn, err, success
    if force == True: priority = 2
    elif out_level == 'QUIET': priority = 0
    elif out_level == 'NORMAL': priority = 1
    elif out_level == 'DEBUG': priority = 2

    if mtype is None and priority>1: 
        print(f'[{stamp}] - ' + '[INFO] ' + msg)
    elif mtype == 'SUCCESS' and priority>0: 
        print(f'[{stamp}] - ' + colored('[SUCCESS] ', 'green') + msg)
    elif mtype == 'WARNING' and priority>0: 
        print(f'[{stamp}] - ' + colored('[WARNING] ', 'yellow') + msg)
    elif mtype == 'ERROR' and priority>0: 
        print(f'[{stamp}] - ' + colored('[ERROR] ', 'red') + msg)