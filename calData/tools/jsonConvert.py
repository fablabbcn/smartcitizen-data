import csv
import json

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