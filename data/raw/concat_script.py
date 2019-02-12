#!/Users/macoscar/anaconda2/bin/python2.7

import pandas as pd
import os
from os.path import dirname, join, realpath
import csv

import argparse

def concatenate(output, index_name, keep, ignore):

	concat = pd.DataFrame()
	raw_src_path = dirname(realpath(__file__))

	print ('Files to concat:')

	header_tokenized = dict()
	for item in os.listdir(raw_src_path):

		if '.csv' in item or '.CSV' in item:
			if item != output and ignore != item:
				print (item)

				src_path = join(raw_src_path, item)
				with open(item, 'r') as csv_file:
					header = csv_file.readlines()[0:4]

				if keep:
					short_tokenized = header[0].strip('\r\n').split(',')
					unit_tokenized = header[1].strip('\r\n').split(',')
					long_tokenized = header[2].strip('\r\n').split(',')
					id_tokenized = header[3].strip('\r\n').split(',')

					for item in short_tokenized:
						if item != '' and item not in header_tokenized.keys():
							index = short_tokenized.index(item)
							
							header_tokenized[short_tokenized[index]] = dict()
							header_tokenized[short_tokenized[index]]['unit'] = unit_tokenized[index]
							header_tokenized[short_tokenized[index]]['long'] = long_tokenized[index]
							header_tokenized[short_tokenized[index]]['id'] = id_tokenized[index]

				temp = pd.read_csv(src_path, verbose=False, skiprows=range(1,4)).set_index("TIME")
				temp.index.rename(index_name, inplace=True)
				concat = concat.combine_first(temp)
	
	columns = concat.columns

	## Sort index
	concat.sort_index(inplace=True)

	name_file = output

	## Save it as CSV
	concat.to_csv(name_file)

	if keep:
		print ('Updating header')
		with open(name_file, 'r') as csv_file:
			content = csv_file.readlines()

			final_header = content[0].strip('\n').split(',')
			short_h = []
			units_h = []
			long_h = []
			id_h = []

			for item in final_header:
				if item in header_tokenized.keys():
					short_h.append(item)
					units_h.append(header_tokenized[item]['unit'])
					long_h.append(header_tokenized[item]['long'])
					id_h.append(header_tokenized[item]['id'])

			content.pop(0)

			for index_content in range(len(content)):
				content[index_content] = content[index_content].strip('\n')

			content.insert(0, ','.join(short_h))
			content.insert(1, ','.join(units_h))
			content.insert(2, ','.join(long_h))
			content.insert(3, ','.join(id_h))

		with open(name_file, 'w') as csv_file:
			print ('Saving file to:', name_file)
			wr = csv.writer(csv_file, delimiter = '\t')
			
			for row in content:
				wr.writerow([row])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", "-o", default = "Log_Concat.csv", help="Output file name, including extension")
	parser.add_argument("--index", "-i", default = "TIME", help="Final name of time index")
	parser.add_argument("--keep", "-k", dest='keep', action='store_true', help="Keep full CSV header")
	parser.add_argument("--ignore", "-ig", dest='ignore', default = "Log_Concat.csv", help="Ignore files in concat")
	parser.set_defaults(keep=False)
	
	args = parser.parse_args()
	concatenate(args.output, args.index, args.keep, args.ignore)

