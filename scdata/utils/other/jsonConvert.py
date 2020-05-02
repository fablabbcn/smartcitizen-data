import csv
import json
import os
        
if os.path.exists('AlphaSense.csv'):
	csvfile = open('AlphaSense.csv', 'r')
	jsonfile = open('AlphaSense.json', 'w')

	fieldnames = ("Serial No", 
		"Zero Current", 
		"Aux Zero Current", 
		"Sensitivity 1", 
		"Sensitivity 2", 
		"Target 1", 
		"Target 2")

	for i in range(2):
		csvfile.next()

	reader = csv.DictReader(csvfile, fieldnames)

	for row in reader:
		json.dump(row, jsonfile)
		jsonfile.write('\n')

if os.path.exists('mics.csv'):
	csvfile = open('mics.csv', 'r')
	jsonfile = open('mics.json', 'w')

	fieldnames = ("Serial No", 
		"Zero Air Resistance 1", 
		"Zero Air Resistance 2", 
		"Sensitivity 1", 
		"Sensitivity 2", 
		"Target 1", 
		"Target 2")

	for i in range(2):
		csvfile.next()

	reader = csv.DictReader(csvfile, fieldnames)

	for row in reader:
		json.dump(row, jsonfile)
		jsonfile.write('\n')