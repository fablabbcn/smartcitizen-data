''' Offline handler used as Device source in unit tests '''
from pandas import DataFrame


class FakeHandler:
    method = 'async'
    # Set by the tests before creating a Device
    sensors = []
    readings = DataFrame()
    failed_sensors = []
    # Blueprint and hardware versions, as SCDevice gives them
    blueprint_url = None
    properties = {}
    channels_by_version = []

    def __init__(self, params):
        self.id = params.id
        self.sensors = list(FakeHandler.sensors)
        self.blueprint_url = FakeHandler.blueprint_url
        self.properties = FakeHandler.properties
        self.channels_by_version = FakeHandler.channels_by_version
        self.timezone = 'UTC'
        self.data = DataFrame()
        self.requested_channels = None
        self.failed_sensors = list(FakeHandler.failed_sensors)

    async def get_data(self, channels=None, **kwargs):
        self.requested_channels = list(channels)
        available = [column for column in FakeHandler.readings.columns if column in channels]
        self.data = FakeHandler.readings[available].copy()
        return self.data

    async def post_data(self, columns=None, rename=None, dry_run=False, **kwargs):
        ''' As SCDevice: posts the columns of its own data (records which ones it had) '''
        self.posted = [column for column in (rename or {}).values() if column in self.data]
        return True

    def apply_blueprint(self, blueprint):
        ''' As SCDevice: the blueprint filled with the hardware (here, as it is) '''
        self.properties = blueprint
        self.channels_by_version = []
        return True
