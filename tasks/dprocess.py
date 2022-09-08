# internal imports
from scdata._config import config
from scdata import Device
from scdata.utils import std_out
import sys

# Config settings
config._out_level = 'DEBUG'
config._timestamp = True
config._avoid_negative_conc = True

def dprocess(device, dryrun = False):
    '''
        This function processes a device from SC API assuming there
        is postprocessing information in it and that it's valid for doing
        so 
    '''    
    std_out(f'[CHUPIFLOW] Processing instance for device {device}')
    # Create device from SC API
    d = Device(descriptor = {'source': 'api', 'id': f'{device}'})
    if d.validate(): 
        # Load only unprocessed
        if d.load(only_unprocessed=True, options = {'resample': False},
            max_amount=config._max_load_amount):
            # Process it
            d.process()
            # Post results
            d.post_metrics(dry_run=dry_run,
                max_retries = config._max_forward_retries)
            # Forward it if requested
            if d.forwarding_request is not None:
                std_out(f'[CHUPIFLOW] Forwarding {device}')
                d.forward(dry_run=dry_run,
                    max_retries = config._max_forward_retries)
            d.update_postprocessing(dry_run=dry_run)
    else:
        std_out(f'[CHUPIFLOW] Device {device} not valid', 'ERROR')
    std_out(f'[CHUPIFLOW] Concluded job for {device}')

if __name__ == '__main__':

    if '-h' in sys.argv or '--help' in sys.argv or '-help' in sys.argv:
        print('dprocess: Process device of SC API')
        print('USAGE:\n\rdprocess.py --device <device-number> [options]')
        print('options:')
        print('--dry-run: dry run')
        sys.exit()
    
    if '--dry-run' in sys.argv: dry_run = True
    else: dry_run = False

    if '--device' in sys.argv:
        device = int(sys.argv[sys.argv.index('--device')+1])
        dprocess(device, dry_run)
