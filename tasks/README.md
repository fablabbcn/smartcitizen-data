# Tasks

Tasks are managed by `chupiflow.py` script and ultimately by `scheduler.py` and CronTab, thanks to `python-crontab` (full doc [here](https://gitlab.com/doctormo/python-crontab)). The script can program tasks in a automated or manual way. If done automatically, it can schedule them `@daily`, `@hourly` and `@minute`, with optional load balancing (not scheduling them all at the same time, but randomly in low load times).

## Start scheduling

This will schedule based on postprocessing information in the platform having a non-null value:

```
python chupiflow.py auto-schedule
```

or (optional dry-run for checks, force-first-run and overwritting if task is already there):

```
python chupiflow.py --dry-run --force-first-run --overwrite
```

Task status and tabfile will be saved in `~/.cache/scdata/tasks` by default. This can be changed in the config:

```
➜  tasks tree -L 2
.
├── 13238
│   └── 13238.log
├── 13486
├── README.md
├── scheduler.log
└── tabfile.tab
```

## Manual scheduling

This will schedule a device regardless the auto-scheduling:

```
python chupiflow.py manual-schedule --device <device> --dry-run --force-first-run --overwrite
```

## Statement

This code was made as a response to the lack of simple task schedulers that are not thought to be running in Azure, Kubernetes or whatever else (`airflow` gets the reference for the name). Nothing against it, just too complex for what could be solved with simple `cron` scheduling.