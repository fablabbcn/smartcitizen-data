#!/usr/bin/python

'''
Example to load device data from a parquet backup on AWS
For this example to work, you need to have the following environment variables set:

# S3 bucket
S3_DATA_BUCKET=
S3_PUBLIC_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
'''

import scdata as sc
device = sc.Device(blueprint='sc_air',
                               params=sc.APIParams(id=18445))

# Load
device.backup_load()

print (device.json)
print (device.data.describe())
