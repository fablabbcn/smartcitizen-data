import pytest
import requests
import json

def test_get_device():
# Device id needs to be as str
    id = 10972
    uuid = '0b648303-edc8-450c-8fc0-99266bf5e0bc'
    url = "https://api.smartcitizen.me/v0/" + "/devices/" + str(id)
    resp = requests.get(url)
    j = json.loads(resp.text)
    assert resp.status_code == 200, resp.text
    assert j['id'] == id, resp.text
    assert j['uuid'] == uuid, resp.text
