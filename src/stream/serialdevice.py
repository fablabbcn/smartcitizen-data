import serial
import serial.tools.list_ports
import time
import sys
import pandas as pd
from src.stream.serialworker import *

class serialdevice:

    def __init__(self, device_type = None, verbose = 2):
        # Serial port
        self.serialPort = None
        self.serialPort_name = None
        self.serialNumber = None
        self.worker = None
        self.verbose = 2     # 0 -> never print anything, 1 -> print only errors, 2 -> print everything
        self.type = device_type

    def set_serial(self):
        device_list = list(serial.tools.list_ports.comports())
        number_devices = len(device_list)
        if number_devices == 0: self.err_out('No device found!!!'); return False
        
        if self.type == 'sck':
            kit_kist = []
            for d in device_list:
                try:
                    if 'Smartcitizen' in d.description:
                        self.std_out('['+str(device_list.index(d))+'] Smartcitizen Kit S/N: ' + d.serial_number)
                        kit_list.append(d)
                except:
                    pass

            number_devices = len(kit_list)
            device_list = kit_list
            if number_devices == 0: self.err_out('No SKC found!!!'); return False                        


        if number_devices == 1:
            which_device = 0
        else:
            for d in device_list: self.std_out(str(device_list.index(d) + 1) + ' --- ' + d.device)
            which_device = int(input('Multiple devices found, please select one: ')) - 1

        self.serialPort_name = device_list[which_device].device
        self.serialNumber = device_list[which_device].serial_number
        return True

    def update_serial(self, speed = 115200, timeout_ser=0.5):
        # Find serial number and assign serial port name
        timeout = time.time() + 15
        while True:
            devList = list(serial.tools.list_ports.comports())
            found = False
            for d in devList:
                try:
                    if self.serialNumber in d.serial_number:
                        self.serialPort_name = d.device
                        found = True
                    if time.time() > timeout:
                        self.err_out('Timeout waiting for device')
                        sys.exit()
                except:
                    pass
            if found: break

        # Open port
        timeout = time.time() + 15
        while self.serialPort is None:
            try:
                time.sleep(0.1)
                self.serialPort = serial.Serial(self.serialPort_name, speed, timeout = timeout_ser)
            except:
                if time.time() > timeout:
                    self.err_out('Timeout waiting for serial port')
                    sys.exit()
            time.sleep(0.1)

    def read_all_serial(self, chunk_size=200):
        """Read all characters on the serial port and return them"""
        if not self.serialPort.timeout:
            raise TypeError('Port needs to have a timeout set!')

        read_buffer = b''

        while True:

            byte_chunk = self.serialPort.read(size=chunk_size)
            read_buffer += byte_chunk
            if not len(byte_chunk) == chunk_size:
                break

        return read_buffer

    def flush(self):
        self.serialPort.reset_input_buffer()

    def start_streaming(self, buffer_length = 10, raster = 0.2, df = pd.DataFrame({'Time': [], 'y': []}, columns = ['Time', 'y'])):
        self.worker = serialworker(self, df, buffer_length, raster)
        self.worker.daemon = True
        self.worker.start()

    def read_line(self):
        return self.serialPort.readline().decode('utf-8').strip('\r\n').split('\t')

    def end(self):
        if self.serialPort.is_open: self.serialPort.close()

    def std_out(self, msg):
        if self.verbose >= 2: print(msg)

    def err_out(self, msg):
        if self.verbose >= 1:
            sys.stdout.write("\033[1;31m")
            print('ERROR ' + msg)
            sys.stdout.write("\033[0;0m")