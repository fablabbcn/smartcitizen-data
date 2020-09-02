from termcolor import colored
from scdata._config import config

def std_out(msg, mtype = None, force = False):
    out_level = config._out_level
    # Output levels:
    # 'QUIET': nothing, 
    # 'NORMAL': warn, err
    # 'DEBUG': info, warn, err, success
    if force == True: priority = 2
    elif out_level == 'QUIET': priority = 0
    elif out_level == 'NORMAL': priority = 1
    elif out_level == 'DEBUG': priority = 2

    if mtype is None and priority>1: 
        print('[INFO]: ' + msg)
    elif mtype == 'SUCCESS' and priority>0: 
        print(colored('[SUCCESS]: ', 'green') + msg)
    elif mtype == 'WARNING' and priority>0: 
        print(colored('[WARNING]: ', 'yellow') + msg)
    elif mtype == 'ERROR' and priority>0: 
        print(colored('[ERROR]: ', 'red') + msg)