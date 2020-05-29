#!/usr/bin/python

import pandas as pd
import os
from os.path import dirname, join, realpath
import csv

def clean():

	concat = pd.DataFrame()
	raw_src_path = dirname(realpath(__file__))

	header_tokenized = dict()
	for item in os.listdir(raw_src_path):
		if '.csv' in item or '.CSV' in item:
				item_name = item[:-4]
				print (item_name)
				print (item)
				clean_file = list()

				src_path = join(raw_src_path, item)
				with open(item, 'r') as csv_file:
					lines = csv_file.readlines()
					for line in lines:
						line = line.strip('\n').split(',')
						print (line)
						print ('-')
						if line [-1] == '':
							print ('cleaning')
							clean_line = line[:-1]
						else:
							clean_line = line
						print (clean_line)
						clean_file.append(clean_line)
				
				with open(join(raw_src_path, item_name + '_CLEAN.csv'), 'w') as csv_file_clean:
					print ('Saving file to:', item_name + '_CLEAN.csv')
					wr = csv.writer(csv_file_clean, delimiter = ',')
			
					for row in clean_file:
						wr.writerow(row)

if __name__ == "__main__":

	clean()

