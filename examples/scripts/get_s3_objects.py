#!/usr/bin/python

'''
Example to load device data from SC and back it up on AWS, including device metadata
For this example to work, you need to have the following environment variables set:

# S3 bucket
S3_DATA_BUCKET=
S3_PUBLIC_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
'''

import asyncio
import json
import os
import time

import boto3
import botocore
import scdata as sc
import structlog

log = structlog.get_logger()

# Get device
device = sc.Device(blueprint='sc_air', params=sc.APIParams(id=18445))
path = 'sandbox'

# Load
log.info('Loading data')
device.options.limit=500
asyncio.run(device.load())

print (device.json)
print (device.data.describe())

log.info('Waiting...')
time.sleep(5)
# Send to storage
device.backup_to_storage(path=path)

# Get data from S3
s3 = boto3.resource('s3')

try:
    metadata = s3.Object(f"{os.environ['S3_DATA_BUCKET']}", f"{path}/{device.id}/metadata.json").get()
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "NoSuchKey":
        # The object does not exist.
        log.warning('Metadata file not found')
    else:
        # Something else has gone wrong.
        log.error(e.response)
except botocore.exceptions.ClientError as e:
    # Something else has gone wrong.
    log.exception(exc_info=e)

response = json.loads(metadata['Body'].read().decode('utf-8'))

log.info("Loaded metadata")
print (response)
