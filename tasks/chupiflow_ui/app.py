#!/usr/bin/env python

from flask import Flask, request, render_template, redirect, url_for
import json
from cron import parsetabfiles, validate, savetabfiles
from extras import get_dpath

app = Flask(__name__)

dpath=None

@app.route('/', methods = ['GET', 'POST'])
def default():
    global dpath
    if request.method == 'POST':
        request.get_data()
        dpath = request.form['path-tab-file']
    else:
        if dpath == '' or dpath is None: dpath=get_dpath()
    tabfiles = parsetabfiles(path=dpath)
    return render_template("jobs.html", tabfiles=tabfiles, defaultpath=dpath)

@app.route('/editjob/<tabfile>-<cron>', methods = ['POST', 'GET'])
def editjob(tabfile,cron,error=None):
    tabfiles=parsetabfiles(path=dpath)

    if request.method == 'POST':
        request.get_data()
        # Form input
        log=request.form["logfile-input"]
        task=request.form["task-input"]
        schedule=request.form["schedule-input"]
        if request.form.getlist("enabled-input") == ['on']: enabled=True 
        else: enabled=False
        who=request.form["who-input"]
        # Validate
        error=validate(schedule, who, task, log)
        tabfiles[tabfile][cron]['logfile']=log
        tabfiles[tabfile][cron]['task']=task
        tabfiles[tabfile][cron]['schedule']=schedule
        tabfiles[tabfile][cron]['enabled']=enabled
        tabfiles[tabfile][cron]['who']=who

        if not error:
            savetabfiles(tabfiles=tabfiles, path=dpath)
            return redirect(url_for("default"))

    crondict=tabfiles[tabfile][cron]
    return render_template("editjob.html", tabfile=tabfile, cron=cron, crondict=crondict, error=error)

@app.route('/tabfiles/<tabfile>')
def tabfile(tabfile):
    tabfiles = parsetabfiles(path=dpath)
    tabpath = f"{dpath}/{tabfile}.tab"
    tab = []
    with open(tabpath, 'r') as file:
        _tab = file.readlines()
    for line in _tab:
        line = line.strip('\n')
        if line != '':
            tab.append(line)
    return render_template("file_viewer.html", file_type='tabfile', file=tab)

@app.route('/logfiles/<tabfile>-<cron>')
def logfile(tabfile, cron):
    tabfiles = parsetabfiles(path=dpath)
    logfile = tabfiles[tabfile][cron]['logfile']
    log = []
    with open(logfile, 'r') as file:
        _log = file.readlines()
    for line in _log:
        line = line.strip('\n')
        if line != '':
            if '[31m' in line:
                line = line.replace('\x1b[31m', '<span style="color:#B22222">')
                line = line.replace('\x1b[0m', '</span>')
            if '[33m' in line:
                line = line.replace('\x1b[33m', '<span style="color:rgb(255,200,0)">')
                line = line.replace('\x1b[0m', '</span>')
            if '[32m' in line:
                line = line.replace('\x1b[32m', '<span style="color:#228B22">')
                line = line.replace('\x1b[0m', '</span>')
            log.append(line)
    return render_template("file_viewer.html", file_type='log', cron=cron, file=log)            

@app.route('/jobfiles/<tabfile>-<cron>')
def taskfile(tabfile, cron):
    tabfiles = parsetabfiles(path=dpath)
    taskfile = tabfiles[tabfile][cron]['task']
    task = []
    with open(taskfile, 'r') as file:
        _task = file.readlines()
    for line in _task:
        line = line.strip('\n')
        if line != '':
            task.append(line)
    return render_template("file_viewer.html", file_type='task', cron=cron, file=task)            

@app.route('/crontabsave', methods=['POST'])
def crontabsave():
    if request.method == 'POST':
        data = json.loads(request.data)
        targetFile = data.pop()
        app.logger.info('%s', targetFile)
        for i in data:
            app.logger.info('%s', i)
            #for line in data:
            #    if validatecron(line):
            #        continue
            #    else:
            #        return(json.dumps({'ERROR':"Wrong cron format: "+line}), 400, {'ContentType':'application/json'})
            with open(cronDir+targetFile,"wb") as fo:
                for line in reversed(data):
                    fo.write(line+"\n")
        return(json.dumps({'success':True}), 200, {'ContentType':'application/json'})
    else:
        return(json.dumps({'ERROR':"Writing file."}), 400, {'ContentType':'application/json'})

if __name__ == '__main__':
   app.run(debug = True)