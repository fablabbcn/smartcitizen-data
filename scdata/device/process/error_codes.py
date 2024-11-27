from enum import Enum
from pandas import DataFrame

class StatusCode(Enum):
    ERROR_MISSING_INPUTS = 'Missing input in kwargs'
    ERROR_CALIBRATION_NOT_FOUND = 'Calibration data not found'
    ERROR_WRONG_CALIBRATION = 'Calibration data does not match'
    ERROR_MISSING_CHANNEL = 'Channels not found'
    ERROR_WRONG_HW = 'Not supported hardware'
    ERROR_UNDEFINED = 'Undefined error'

    WARNING_EMPTY_ID = 'Calibration ID is empty'
    WARNING_NULL_CHANNEL = 'Channel name is null'

    SUCCESS = 'Success'

    DEFAULT = 'Default processing code'

class ProcessResult():
    data: DataFrame = None
    status_code: StatusCode = None

    def __init__(self, data = None, code=StatusCode.DEFAULT):
        self.data = data
        self.status_code = code