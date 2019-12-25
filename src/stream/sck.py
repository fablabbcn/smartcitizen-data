import os
import subprocess
from src.stream.uf2.utils import uf2conv
import shutil
import binascii
import json
import requests
import traceback
import pandas as pd
from src.stream.serialdevice import *

'''
Smartcitizen Kit python library.
This library is meant to be run inside the firmware repository folder.
'''

class sck:

    def __init__(self, verbose = 2):
        serialdevice.__init__('sck')
        self.sensors = []
        self.verbose = verbose
        # self.updateSerial()

    # paths
    # paths = {}
    # paths['base'] = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip()
    # paths['binFolder'] = os.path.join(str(paths['base']), 'bin')
    # paths['esptoolPy'] = os.path.join(str(paths['base']), 'tools', 'esptool.py')
    # os.chdir('esp')
    # paths['pioHome'] = [s.split()[1].strip(',').strip("'") for s in subprocess.check_output(['pio', 'run', '-t', 'envdump']).split('\n') if "'PIOHOME_DIR'" in s][0]
    # os.chdir(paths['base'])
    # paths['esptool'] = os.path.join(str(paths['pioHome']), 'packages', 'tool-esptool', 'esptool')

    # # filenames
    # files = {}
    # files['samBin'] = 'SAM_firmware.bin'
    # files['samUf2'] = 'SAM_firmware.uf2'
    # files['espBin'] = 'ESP_firmware.bin'

    # # Serial port
    # serialPort = None
    # serialPort_name = None

    # chips and firmware info
    infoReady = False
    sam_serialNum = ''
    sam_firmVer = ''
    sam_firmCommit = ''
    sam_firmBuildDate = ''
    esp_macAddress = ''
    esp_firmVer = ''
    esp_firmCommit = ''
    esp_firmBuildDate = ''

    # WiFi and platform info
    mode = ''
    token = ''
    wifi_ssid = ''
    wifi_pass = ''

    verbose = 2     # 0 -> never print anything, 1 -> print only errors, 2 -> print everything

    def begin(self):
        if self.set_serial(device_type = 'sck'):
            for retry in range(3):
                if self.getSensors(): break
                return False

        # devList = list(serial.tools.list_ports.comports())
        # kit_list = []
        # for d in devList:
        #     try:
        #         if 'Smartcitizen' in d.description:
        #             self.std_out('['+str(i)+'] Smartcitizen Kit S/N: ' + d.serial_number)
        #             kit_list.append(d)
        #     except:
        #         pass
        # number_kits = len(kit_list)
        # if number_kits == 0: self.err_out('No SKC found!!!'); return False
        # elif number_kits == 1:
        #     which_kit = 0
        # else:
        #     which_kit = int(raw_input('Multiple Kits found, please select one: ')) - 1

        # self.sam_serialNum = kit_list[which_kit].serial_number
        # self.serialPort_name = kit_list[which_kit].device

        return True  

    # def updateSerial(self, speed = 115200, timeout_ser=0.5):
    #     timeout = time.time() + 15
    #     while True:
    #         devList = list(serial.tools.list_ports.comports())
    #         found = False
    #         for d in devList:
    #             try:
    #                 if self.sam_serialNum in d.serial_number:
    #                     self.serialPort_name = d.device
    #                     found = True
    #                 if time.time() > timeout:
    #                     self.err_out('Timeout waiting for device')
    #                     sys.exit()
    #             except:
    #                 pass
    #         if found: break

    #     timeout = time.time() + 15
    #     while self.serialPort is None:
    #         try:
    #             time.sleep(0.1)
    #             self.serialPort = serial.Serial(self.serialPort_name, speed, timeout = timeout_ser)
    #             self.serialPort.set_buffer_size(rx_size = 5000000, tx_size = 5000000)
    #         except:
    #             if time.time() > timeout:
    #                 self.err_out('Timeout waiting for serial port')
    #                 sys.exit()
    #         time.sleep(0.1)
    #         try:
    #             if self.serialPort.write('\r\n'): return
    #         except:
    #             pass

    def checkConsole(self):
        timeout = time.time() + 15
        while True:
            self.serialPort.write('\r\n'.encode())
            time.sleep(0.1)
            # buff = self.serialPort.read(self.serialPort.in_waiting).decode("utf-8")
            buff = self.read_all_serial(chunk_size=200).decode('utf-8')
            if 'SCK' in buff: return True
            if time.time() > timeout:
                self.err_out('Timeout waiting for kit console response')
                return False
            time.sleep(0.5)

    def getInfo(self):
        if self.infoReady: return
        self.updateSerial()
        self.serialPort.write('\r\nversion\r\n'.encode())
        time.sleep(0.5)
        m = self.read_all_serial(chunk_size=200).decode('utf-8')
        self.esp_macAddress = m[m.index('address:')+1]
        m.remove('SAM')
        self.sam_firmVer = m[m.index('SAM')+2]
        m.remove('ESP')
        self.esp_firmVer = m[m.index('ESP')+2]
        self.infoReady = True

    # def read_all_serial(self, chunk_size=200):
    #     """Read all characters on the serial port and return them."""
    #     if not self.serialPort.timeout:
    #         raise TypeError('Port needs to have a timeout set!')

    #     read_buffer = b''

    #     while True:
    #         # Read in chunks. Each chunk will wait as long as specified by
    #         # timeout. Increase chunk_size to fail quicker
    #         byte_chunk = self.serialPort.read(size=chunk_size)
    #         read_buffer += byte_chunk
    #         if not len(byte_chunk) == chunk_size:
    #             break

    #     return read_buffer

    def getConfig(self):
        self.updateSerial()
        self.checkConsole()
        self.serialPort.write('\r\nconfig\r\n'.encode())
        time.sleep(0.5)
        m = self.read_all_serial(chunk_size=200).decode('utf-8')
        for line in m:
            if 'Mode' in line:
                mm = line.split('Mode: ')[1].strip()
                if mm != 'not configured': self.mode = mm
            if 'Token:' in line:
                tt = line.split(':')[1].strip()
                if tt != 'not configured' and len(tt) == 6: self.token = tt
            if 'credentials:' in line:
                ww = line.split('credentials: ')[1].strip()
                if ww.count(' - ') == 1:
                    self.wifi_ssid, self.wifi_pass = ww.split(' - ')
                    if self.wifi_pass == 'null': self.wifi_pass = ""

    def getSensors(self):

        self.updateSerial()
        self.checkConsole()
        self.serialPort.write('sensor\r\n'.encode())

        m = self.read_all_serial(chunk_size=200).decode("utf-8").split('\r\n')
        
        while '----------' in m: m.remove('----------')
        while 'SCK > ' in m: m.remove('SCK > ')

        self.sensor_enabled = dict()
        if 'Enabled' in m:
            for key in m[m.index('Enabled')+1:]:
                name = key[:key.index('(')]
                self.sensor_enabled[name[:-1]] = key[key.index('(') + 1:-1] 
            self.sensor_disabled = m[m.index('Disabled')+1:m.index('Enabled')]

            return True
        else:
            return False

    def enableSensor(self, sensor):
        self.updateSerial()  
        self.checkConsole()
        self.getSensors()

        if sensor in self.sensor_enabled.keys(): 
            self.std_out('Sensor already enabled', 'WARNING')
            return True

        else:
            self.std_out('Enabling sensor ' + sensor)
            command = 'sensor -enable ' + sensor + '\r\n'
            self.serialPort.write(command.encode())
            
            self.getSensors()
            print (self.sensor_enabled.keys())
            if sensor in self.sensor_enabled.keys(): return True
            else: return False

    def disableSensor(self, sensor):
        self.updateSerial()  
        self.checkConsole()
        self.getSensors()

        if sensor in self.sensor_enabled.keys(): 
            self.std_out('Sensor already enabled', 'WARNING')
            return True

        else:
            self.std_out('Disabling sensor ' + sensor)
            command = 'sensor -disable ' + sensor + '\r\n'
            self.serialPort.write(command.encode())
            
            self.getSensors()
            if sensor in self.sensor_disabled: return True
            else: return False

    def toggleShell(self):
        self.updateSerial()
        self.checkConsole()

        if not self.statusShell():
            self.std_out('Setting shell mode')
            command = '\r\nshell -on\r\n'
            self.serialPort.write(command.encode())
        else:
            self.std_out('Setting normal mode')
            command = '\r\nshell -off\r\n'
            self.serialPort.write(command.encode())

    def statusShell(self):
        self.updateSerial()
        self.checkConsole()

        self.serialPort.write('shell\r\n'.encode())
        time.sleep(0.5)
        m = self.read_all_serial().decode("utf-8").split('\r\n')
        for line in m:
            if 'Shell mode' in line:
                if 'off' in line: 
                    return False
                if 'on' in line: 
                    return True

    def monitor(self, sensors = None, noms = True, notime = False, sd = False):
        self.updateSerial()
        self.checkConsole()
        self.getSensors()

        command = 'monitor '
        if noms: command = command + '-noms '
        if notime: command = command + '-notime '
        if sd: command = command + '-sd '
        
        if type(sensors) != list: sensors = sensors.split(',')
        if sensors is not None:
            for sensor in sensors:
                if sensor not in self.sensor_enabled: 
                    if not self.enableSensor(sensor): 
                        self.err_out(f'Cannot enable {sensor}')
                        return False
                command = command + sensor + ', '
            command = command + '\n'

        self.serialPort.write(command.encode())
        self.serialPort.readline()

        # Get columns
        columns = self.read_line()
        df_empty = dict()
        for column in columns: df_empty[column] = []
        # if not notime: 
        df = pd.DataFrame(df_empty, columns = columns)
            # df.set_index('Time', inplace = True)
            # columns.remove('Time')

        self.start_streaming(df)

    def setBootLoaderMode(self):
        self.updateSerial()
        self.serialPort.close()
        self.serialPort = serial.Serial(self.serialPort_name, 1200)
        self.serialPort.setDTR(False)
        time.sleep(5)
        mps = uf2conv.getdrives()
        for p in mps:
            if 'INFO_UF2.TXT' in os.listdir(p):
                return p
        self.err_out('Cant find the mount point fo the SCK')
        return False

    def buildSAM(self, out=sys.__stdout__):
        os.chdir(self.paths['base'])
        os.chdir('sam')
        piorun = subprocess.call(['pio', 'run'], stdout=out, stderr=subprocess.STDOUT)
        if piorun == 0:
            try:
                shutil.copyfile(os.path.join(os.getcwd(), '.pioenvs', 'sck2', 'firmware.bin'), os.path.join(self.paths['binFolder'], self.files['samBin']))
            except:
                self.err_out('Failed building SAM firmware')
                return False
        with open(os.path.join(self.paths['binFolder'], self.files['samBin']), mode='rb') as myfile:
            inpbuf = myfile.read()
        outbuf = uf2conv.convertToUF2(inpbuf)
        uf2conv.writeFile(os.path.join(self.paths['binFolder'], self.files['samUf2']), outbuf)
        os.chdir(self.paths['base'])
        return True

    def flashSAM(self, out=sys.__stdout__):
        os.chdir(self.paths['base'])
        mountpoint = self.setBootLoaderMode()
        try:
            shutil.copyfile(os.path.join(self.paths['binFolder'], self.files['samUf2']), os.path.join(mountpoint, self.files['samUf2']))
        except:
            self.err_out('Failed transferring firmware to SAM')
            return False
        time.sleep(2)
        return True

    def getBridge(self, speed=921600):
        timeout = time.time() + 15
        while True:
            self.updateSerial(speed)
            self.serialPort.write('\r\n'.encode())
            time.sleep(0.1)
            buff = self.read_all_serial(chunk_size=200).decode('utf-8')
            if 'SCK' in buff: break
            if time.time() > timeout:
                self.err_out('Timeout waiting for SAM bridge')
                return False
            time.sleep(2.5)
        buff = self.serialPort.read(self.serialPort.in_waiting)
        self.serialPort.write('esp -flash ' + str(speed) + '\r\n')
        time.sleep(0.2)
        buff = self.serialPort.read(self.serialPort.in_waiting)
        return True

    def buildESP(self, out=sys.__stdout__):
        os.chdir(self.paths['base'])
        os.chdir('esp')
        piorun = subprocess.call(['pio', 'run'], stdout=out, stderr=subprocess.STDOUT)
        if piorun == 0:
            shutil.copyfile(os.path.join(os.getcwd() , '.pioenvs', 'esp12e', 'firmware.bin'), os.path.join(self.paths['binFolder'], self.files['espBin']))
            return True
        self.err_out('Failed building ESP firmware')
        return False

    def flashESP(self, speed=921600, out=sys.__stdout__):
        os.chdir(self.paths['base'])
        if not self.getBridge(speed): return False
        flashedESP = subprocess.call([self.paths['esptool'], '-cp', self.serialPort_name, '-cb', str(speed), '-ca', '0x000000', '-cf', os.path.join(self.paths['binFolder'], self.files['espBin'])], stdout=out, stderr=subprocess.STDOUT)
        if flashedESP == 0:
            time.sleep(1)
            return True
        else:
            self.err_out('Failed transferring ESP firmware')
            return False

    def eraseESP(self):
        if not self.getBridge(): return False
        flashedESPFS = subprocess.call([self.paths['esptoolPy'], '--port', self.serialPort_name, 'erase_flash'], stderr=subprocess.STDOUT)
        if flashedESPFS == 0:
            time.sleep(1)
            return True
        else: return False

    def reset(self):
        self.updateSerial()
        self.checkConsole();
        self.serialPort.write('\r\n')
        self.serialPort.write('reset\r\n')

    def netConfig(self):
        if len(self.wifi_ssid) == 0 or len(self.token) != 6:
            print('WiFi and token MUST be set!!')
            return False
        self.updateSerial()
        self.checkConsole()
        command = '\r\nconfig -mode net -wifi "' + self.wifi_ssid + '" "' + self.wifi_pass + '" -token ' + self.token + '\r\n'
        self.serialPort.write(command.encode())
        # TODO verify config success
        return True

    def sdConfig(self):
        self.updateSerial()
        self.checkConsole();
        command = '\r\ntime ' + str(int(time.time())) + '\r\n'
        self.serialPort.write(command.encode())
        if len(self.wifi_ssid) == 0:
            self.serialPort.write('config -mode sdcard\r\n'.encode())
        else:
            command = 'config -mode sdcard -wifi "' + self.wifi_ssid + '" "' + self.wifi_pass + '"\r\n'
            self.serialPort.write(command.encode())
        # TODO verify config success
        return True

    def resetConfig(self):
        self.updateSerial()
        self.checkConsole();
        self.serialPort.write('\r\nconfig -defaults\r\n'.encode())
        # TODO verify config success
        return True

    def register(self):
        try:
            import secret
            print("Found secrets.py:")
            print("bearer: " + secret.bearer)
            print("Wifi ssid: " + secret.wifi_ssid)
            print("Wifi pass: " + secret.wifi_pass)
            bearer = secret.bearer
            wifi_ssid = secret.wifi_ssid
            wifi_pass = secret.wifi_pass
        except:
            bearer = raw_input("Platform bearer: ")
            wifi_ssid = raw_input("WiFi ssid: ")
            wifi_pass = raw_input("WiFi password: ")

        headers = {'Authorization':'Bearer ' + bearer, 'Content-type': 'application/json',}
        device = {}
        try:
            device['name'] = self.platform_name
        except:
            print('Your device needs a name!')
            # TODO ask for a name
            sys.exit()
        device['device_token'] = binascii.b2a_hex(os.urandom(3))
        self.token = device['device_token']
        device['description'] = ''
        device['kit_id'] = 20
        device['latitude'] = 41.396867
        device['longitude'] = 2.194351
        device['exposure'] = 'indoor'
        device['user_tags'] = 'Lab, Research, Experimental'

        device_json = json.dumps(device)
        backed_device = requests.post('https://api.smartcitizen.me/v0/devices', data=device_json, headers=headers)
        self.id = str(backed_device.json()['id'])
        self.platform_url = "https://smartcitizen.me/kits/" + self.id

        self.serialPort.write('\r\nconfig -mode net -wifi "' + wifi_ssid + '" "' + wifi_pass + '" -token ' + self.token + '\r\n')
        time.sleep(1)

    def inventory_add(self):

        self.getInfo()

        if not hasattr(self, 'token'):
            self.token = ''
        if not hasattr(self, 'platform_name'):
            self.platform_name = ''
        if not hasattr(self, 'platform_url'):
            self.platform_url = ''

        inv_path = "inventory.csv"

        if os.path.exists(inv_path):
            shutil.copyfile(inv_path, inv_path+".BAK")
            csvFile = open("inventory.csv", "a")
        else:
            csvFile = open(inv_path, "w")
            # time,serial,mac,sam_firmVer,esp_firmVer,description,token,platform_name,platform_url
            csvFile.write("time,serial,mac,sam_firmVer,esp_firmVer,description,token,platform_name,platform_url\n")

        csvFile.write(time.strftime("%Y-%m-%dT%H:%M:%SZ,", time.gmtime()))
        csvFile.write(self.sam_serialNum + ',' + self.esp_macAddress + ',' + self.sam_firmVer + ',' + self.esp_firmVer + ',' + self.description + ',' + self.token + ',' + self.platform_name + ',' + self.platform_url + '\n')
        csvFile.close()