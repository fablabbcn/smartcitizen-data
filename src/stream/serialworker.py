import pandas as pd
import multiprocessing
from multiprocessing import Queue
from datetime import datetime
import numpy as np
import time

class serialworker(multiprocessing.Process):

	def __init__ (self, device, df, buffer_length = 10, raster = 0.2, verbose = True):

		multiprocessing.Process.__init__(self)
		self.input = Queue()
		self.output = Queue()
		self.device = device
		self.columns = list(df.columns.values)
		self.dataframe = df
		self.example = df.set_index('Time')
		self.buffer_length = buffer_length
		self.verbose = verbose
		self.raster = raster

		self.std_out(f'Initialised serial worker for device on port {self.device.serialPort_name}. Buffering {self.buffer_length} samples')

	def std_out(self, msg):
		if self.verbose: print (msg)

	def run(self):
		count_buffer = 0
		self.device.flush()
		last = datetime.now()
		
		while True:
			if not self.input.empty():
				self.std_out('Terminating serialworker')
				task = self.input.get()
				
				if task == "stop": 
					self.terminate()
					self.join()
					time.sleep(0.1)
					if not self.is_alive(): 
						self.join(timeout=1.0)
						self.std_out('Time out set to 1')
						self.input.close()
						break
			
			now = datetime.now()
			data = self.device.read_line()
			if (now - last).total_seconds() > self.raster:
				last = now
				if 'Time' in self.columns: 

					if len(data) < len (self.columns): data.insert(0, pd.to_datetime(now))
					else: data[0] = pd.to_datetime(data[0])
					
					try: 
						data[1:] = list(map(float, data[1:]))
					except:
						data[1:] = [np.nan]
						pass

					df_stream = pd.DataFrame([data[:]], columns = self.columns)
				
				self.dataframe = pd.concat([self.dataframe, df_stream], sort=False)
				count_buffer += 1
				if count_buffer == self.buffer_length: 
					self.output.put(self.dataframe)
					count_buffer = 0
					self.dataframe = self.example