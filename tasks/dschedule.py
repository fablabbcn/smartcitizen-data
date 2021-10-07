from os.path import join
from os import makedirs
import sys

from scdata._config import config
from scdata.utils import std_out
from scdata.io.device_api import ScApiDevice
from scdata import Device

config._out_level = 'DEBUG'
config._timestamp = True

from scheduler import Scheduler

def dschedule(interval_hours, dry_run = False):
    '''
        This function schedules processing SC API devices based
        on the result of a global query for data processing
        in the SC API
    '''
    try:
        df = ScApiDevice.search_by_query(key="postprocessing_id",
                                         value="not_null", full= True)
    except:
        pass
        return None

    # Check devices to postprocess first
    dl = []

    for device in df.index:
        std_out(f'[CHUPIFLOW] Checking postprocessing for {device}')
        scd = Device(descriptor={'source': 'api', 'id': device})
        # Avoid scheduling invalid devices
        if scd.validate(): dl.append(device)
        else: std_out(f'[CHUPIFLOW] Device {device} not valid', 'ERROR')

    for d in dl:
        # Set scheduler
        s = Scheduler()
        # Define task
        task = f'{config._device_processor}.py --device {d}'
        #Create log output if not existing
        dt = join(config.paths['tasks'], str(d))
        makedirs(dt, exist_ok=True)
        log = f"{join(dt, f'{config._device_processor}_{d}.log')}"
        # Schedule task
        s.schedule_task(task = task,
                        log = log,
                        interval = f'{interval_hours}H',
                        dry_run = dry_run,
                        load_balancing = True)

if __name__ == '__main__':

    if '-h' in sys.argv or '--help' in sys.argv or '-help' in sys.argv:
        print('dschedule: Schedule tasks for devices to process in SC API')
        print('USAGE:\n\rdschedule.py [options]')
        print('options:')
        print('--interval-hours: taks execution interval in hours (default: scdata.config._postprocessing_interval_hours)')
        print('--dry-run: dry run')
        sys.exit()

    if '--dry-run' in sys.argv: dry_run = True
    else: dry_run = False

    if '--interval-hours' in sys.argv:
        interval = int(sys.argv[sys.argv.index('--interval-hours')+1])
    else:
        interval = config._postprocessing_interval_hours

    dschedule(interval, dry_run)
