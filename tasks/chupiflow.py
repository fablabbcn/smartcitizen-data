import sys
from os.path import join
from os import makedirs

if __name__ == '__main__':

    if '-h' in sys.argv or '--help' in sys.argv:
        print('chupiflow: Process device of SC API')
        print('USAGE:\n\rchupiflow.py [options] action')
        print('options:')
        print('--dry-run: dry run')
        print('--force-first-run: force first time running job')
        print('--overwrite: overwrite if it exists already')
        print('actions: auto-schedule or device-schedule')
        print('auto-schedule --interval-days <interval-days> (config._postprocessing_interval_hours):')
        print('\tschedule devices postproccesing check based on device postprocessing in platform')
        print('\tauto-schedule makes a global task for checking on interval-days interval and then the actual tasks are scheduled based on default intervals')
        print('manual-schedule --device <device> --interval-hours <interval-hours> (config._postprocessing_interval_hours):') 
        print('\tschedule device processing manually')
        sys.exit()

    from scheduler import Scheduler
    from scdata._config import config    

    if '--dry-run' in sys.argv: dry_run = True
    else: dry_run = False

    if '--force-first-run' in sys.argv: force_first_run = True
    else: force_first_run = False

    if '--overwrite' in sys.argv: overwrite = True
    else: overwrite = False

    if 'auto-schedule' in sys.argv:
        if '--interval-days' in sys.argv: 
            interval = int(sys.argv[sys.argv.index('--interval-days')+1])
        else: 
            interval = config._scheduler_interval_days

        s = Scheduler()
        s.schedule_task(task = f'{config._device_scheduler}.py', 
                        log = join(config.paths['tasks'], config._scheduler_log), 
                        interval = f'{interval}D',
                        force_first_run = force_first_run,
                        overwrite = overwrite,
                        dry_run = dry_run)
        sys.exit()

    if 'manual-schedule' in sys.argv:
        if '--device' not in sys.argv:
            print ('Cannot process without a devide ID')
            sys.exit()
        if '--interval-hours' in sys.argv: 
            interval = int(sys.argv[sys.argv.index('--interval-hours')+1])
        else: 
            interval = config._postprocessing_interval_hours
        # Setup scheduler
        s = Scheduler()
        device = int(sys.argv[sys.argv.index('--device')+1])
        dt = join(config.paths['tasks'], str(device))
        makedirs(dt, exist_ok=True)
        
        s.schedule_task(task = f'{config._device_processor}.py --device {device}', 
                        log = join(dt, f'{device}.log'),
                        interval = f'{interval}H',
                        force_first_run = force_first_run,
                        overwrite = overwrite,
                        dry_run = dry_run)
        sys.exit()