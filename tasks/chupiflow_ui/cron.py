from os import listdir
from os.path import join
import traceback
import sys
import subprocess
from crontab import CronTab
import threading

class CronThread(threading.Thread):
    def __init__(self, job):
        self.job = job
        super().__init__()
        self.status = 'init'

    def run(self):
        self.status = 'running'
        self.job.run()
        self.status = 'done'

def parsetabfiles(path):
    tabfiles = {}
    try:
        for tabfile in listdir(path):
            if tabfile.endswith('.tab'):
                tname = tabfile.replace('.tab', '')
                tabfiles[tname]=dict()
                jobs=CronTab(tabfile=join(path, tabfile))

                for job in jobs:
                    tabfiles[tname][job.comment]=dict()
                    tabfiles[tname][job.comment]['schedule']=job.slices
                    tabfiles[tname][job.comment]['enabled']=job.is_enabled()
                    tabfiles[tname][job.comment]['valid']=job.is_valid()
                    cl = job.command.split(' ')
                    tabfiles[tname][job.comment]['who']=cl[0]
                    tabfiles[tname][job.comment]['task']=' '.join(cl[1: cl.index('>>')])
                    tabfiles[tname][job.comment]['logfile']=cl[cl.index('>>')+1:-1][0]
        return tabfiles
    except IOError:
        traceback.print_exc()
        pass
    return "Unable to read file"

def validate(schedule, who, task, log):
    c=CronTab(user=True)

    command = f"{who} {task} >> {log} 2>&1"
    j=c.new(command=command)

    if not j.is_valid(): return 'Command error'

    try:
        j.setall(schedule)
    except ValueError:
        pass
        return 'Time slice error'

    return None

def triggercrontab(path,tabfile,cron):
    print (f'Triggering {cron} from {tabfile}')
    jobs=CronTab(tabfile=join(path, tabfile+'.tab'))

    for job in jobs:
        if job.comment==cron:
            if job.is_valid:
                ct = CronThread(job)
                ct.start()
                return ct
            else:
                return False

def savetabfiles(tabfiles, path):
    for tabfile in tabfiles:
        output = []
        for job in tabfiles[tabfile]:
            if tabfiles[tabfile][job]['enabled']: enable = ''
            else: enable = '# '
            line = f"{enable}{tabfiles[tabfile][job]['schedule']} {sys.executable} {tabfiles[tabfile][job]['task']} >> {tabfiles[tabfile][job]['logfile']} 2>&1 # {job}"
            output.append(line)

        outputfile=open(join(path, f"{tabfile}.tab"),'w')
        for line in output:
            outputfile.write(line)
            outputfile.write('\n')
        outputfile.close()
        subprocess.call(['crontab', join(path, f"{tabfile}.tab")])
    return 'Mierda de edit que has hecho'