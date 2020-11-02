from requests import get
from scdata.utils import std_out, localise_date
from os import environ

from pandas import DataFrame

class ScApiUser:

    API_BASE_URL = 'https://api.smartcitizen.me/v0/users/'
    headers = {'Authorization':'Bearer ' + environ['SC_BEARER'], 'Content-type': 'application/json'}

    def __init__ (self, did = None, username = None):
        self.id = did
        self.username = username
        self.devices = None
        self.userjson = None
        self.location = None
        self.url = None
        self.joined_at = None
        self.updated_at = None

    def get_user_info(self):
        if self.id is None and self.username is None:
            std_out('Need at lease username or user id to make a valid request')

        if self.id is not None: self.get_user_json_by_id()
        if self.username is not None: self.get_user_json_by_username()

        try:
            self.devices = self.userjson['devices']
            self.location = self.userjson['location']

            self.joined_at = self.userjson['joined_at']
            self.updated_at = self.userjson['updated_at']
        except:
            std_out('Problem while getting user info', 'ERROR')
            pass
        else: 
            return True

        return False

    def get_user_json_by_id(self):
        if self.userjson is None:
            try:
                userR = get(self.API_BASE_URL + '{}/'.format(self.id), headers = self.headers)
                print (userR)
                if userR.status_code == 200 or userR.status_code == 201:
                    self.userjson = userR.json()
                else: 
                    std_out('API reported {}'.format(userR.status_code), 'ERROR')  
            except:
                std_out('Failed request. Probably no connection', 'ERROR')  
                pass                
        return self.userjson

    def get_user_json_by_username(self):
        if self.userjson is None:
            try:
                userR = get(self.API_BASE_URL + '{}/'.format(self.username), headers = self.headers)
                print (userR)
                if userR.status_code == 200 or userR.status_code == 201:
                    self.userjson = userR.json()
                else: 
                    std_out('API reported {}'.format(userR.status_code), 'ERROR')  
            except:
                std_out('Failed request. Probably no connection', 'ERROR')  
                pass                
        return self.userjson