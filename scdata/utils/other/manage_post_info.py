from requests import patch
import json
from scdata._config import config
from os import environ
import datetime
import hashlib
import argparse

def create_post_info(kit_id, hardware_id):
    headers = {'Authorization':'Bearer ' + environ['SC_BEARER'], 'Content-type': 'application/json'}

    post_info = {
    				"postprocessing_info": 
    				{
    					"updated_at": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
    					"template_version_hash": hashlib.sha256(hardware_id.encode()).hexdigest(),
    					"latest_postprocessing": None, 
    					"template_url": config.hardware_url
    				}
    			}

    # Example postprocessing_info:
	# {
	# 	"updated_at": "2020-10-29T04:35:23Z",
	# 	"template_url": "https://github.com/fablabbcn/smartcitizen-data/blob/master/hardware/hardware.json",
	# 	"template_version_hash": "434a1d245c3f115e99988a4ae3ed6751bb9f31c3",
	# 	"latest_postprocessing": "2020-10-29T08:35:23Z"
	# }    			

    post_json = json.dumps(post_info)
    print (f'Posted request info:\n{post_json}')
    response = patch(f'https://api.smartcitizen.me/v0/devices/{kit_id}/', data = post_json, headers = headers)

    print (f'Patch url: {response.url}')
    print (f'Request response status: {response.status_code}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--did", "-d", help="Device ID in SC platform to manage data")
    # https://github.com/fablabbcn/smartcitizen-data/blob/master/hardware/hardware.json
    parser.add_argument("--hid", "-hw", help="Hardware ID of the physical device. Must be in hardware.json")
    # parser.add_argument("--input", "-i", help="json file containing hardware id and sensor ids")

    args = parser.parse_args()
    create_post_info(args.did, args.hid) 