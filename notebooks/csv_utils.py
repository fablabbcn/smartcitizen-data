# All here should go under 1.1.2

start_date = 0
end_date = 0
openData = list()

def _upload():
    
    _upload_widget = fileupload.FileUploadWidget()
    _tz_widget = widgets.Dropdown(options=pytz.common_timezones, value='UTC', description='Timezone: ')
    
    def _cb(change):
        # get file
        decoded = io.StringIO(change['owner'].data.decode('utf-8'))
        filename = change['owner'].filename 
        fileData = io.StringIO(change['new'].decode('utf-8'))
        df = pd.read_csv(fileData, verbose=True, skiprows=[1]).set_index('Time')
          
        # prepare dataframe
        print df.index
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert(_tz_widget.value)
        df.sort_index(inplace=True)
        df = df.groupby(pd.TimeGrouper(freq='10Min')).aggregate(np.mean)
        df.drop([i for i in df.columns if 'Unnamed' in i], axis=1, inplace=True)
        
        readings[filename] = df[df.index > '2001-01-01T00:00:01Z']
        if start_date > 0: readings[filename] = df[df.index > start_date]
        if end_date > 0: readings[filename] = df[df.index < end_date]
        listFiles(filename)
    
    # widgets
    _upload_widget.observe(_cb, names='data')
    _hb = widgets.HBox([_upload_widget, _tz_widget, widgets.HTML(' ')])
    display(_hb)

def delFile(b):
    clear_output()
    for d in list(b.hbl.children): d.close()
    readings.pop(b.f)

def describeFile(b):
    clear_output()
    display(readings[b.f].describe())
    
def exportFile(b):
    export_dir = 'exports'
    if not os.path.exists(export_dir): os.mkdir(export_dir)
    savePath = os.path.join(export_dir, b.f+'_clean_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H%M%S')
+'.csv')
    if not os.path.exists(savePath):
        readings[b.f].to_csv(savePath, sep=",")
        display(FileLink(savePath))
    else:
        display(widgets.HTML(' File Already exists!'))
    
def listFiles(filename):
#     clear_output()
    temp = list(fileList.children)
    cb = widgets.Button(icon='close',layout=widgets.Layout(width='30px'))
    cb.on_click(delFile)
    cb.f = filename
    eb = widgets.Button(description='Export processed CSV', layout=widgets.Layout(width='180px'))
    eb.on_click(exportFile)
    eb.f = filename
    sb = widgets.Button(description='describe', layout=widgets.Layout(width='80px'))
    sb.on_click(describeFile)
    sb.f = filename  
    hbl = widgets.HBox([cb, widgets.HTML(' <b>'+filename+'</b> \t'), sb, eb])
    cb.hbl = hbl
    temp.append(hbl)
    fileList.children = temp

readings = {}
display(widgets.HTML('<hr><h3>Select CSV files (remember to change the timezone!)</h3>'))
_upload()
fileList = widgets.VBox([widgets.HTML('<hr>')])
display(fileList)
