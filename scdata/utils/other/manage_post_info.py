from requests import patch
import json
from scdata._config import config
from os import environ
import datetime
import hashlib
import argparse

def create_post_info(kit_id, hardware_url, blueprint_url):
    headers = {'Authorization':'Bearer ' + environ['SC_BEARER'], 'Content-type': 'application/json'}

    post_info = {
                    "postprocessing_info": 
                    {
                        "updated_at": datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "blueprint_url": blueprint_url,
                        "hardware_url": hardware_url,
                        "latest_postprocessing": None, 
                    }
                }

    # Example postprocessing_info:
    # {
    #   "updated_at": "2020-10-29T04:35:23Z",
    #   "blueprint_url": "https://github.com/fablabbcn/smartcitizen-data/blob/master/blueprints/sc_21_station_module.json",
    #   "hardware_url": "https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/master/hardware/SCAS210001.json",
    #   "latest_postprocessing": "2020-10-29T08:35:23Z"
    # }             

    post_json = json.dumps(post_info)
    print (f'Posted request info:\n{post_json}')
    response = patch(f'https://api.smartcitizen.me/v0/devices/{kit_id}/', data = post_json, headers = headers)

    print (f'Patch url: {response.url}')
    print (f'Request response status: {response.status_code}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--did", "-d", help="Device ID in SC platform to manage data")
    parser.add_argument("--hardware_url", "-hw", help="Hardware url json description file")
    parser.add_argument("--blueprint_url", "-b", help="Post processing blueprint url json description_file")

    args = parser.parse_args()
    create_post_info(args.did, args.hardware_url, args.blueprint_url)
