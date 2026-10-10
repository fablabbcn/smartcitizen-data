from enum import Enum
from pandas import DataFrame

class StatusCode(Enum):
    ERROR_MISSING_INPUTS = 'Missing input in kwargs'
    ERROR_CALIBRATION_NOT_FOUND = 'Calibration data not found'
    ERROR_WRONG_CALIBRATION = 'Calibration data does not match'
    ERROR_MISSING_CHANNEL = 'Channels not found'
    ERROR_WRONG_HW = 'Not supported hardware'
    ERROR_WRONG_INDEX = 'Index type not supported'
    ERROR_UNDEFINED = 'Undefined error'

    WARNING_EMPTY_ID = 'Calibration ID is empty'
    WARNING_NULL_CHANNEL = 'Channel name is null'

    SUCCESS = 'Success'

    DEFAULT = 'Default processing code'

class ProcessResult():
    data: DataFrame = None
    status_code: StatusCode = None
    # Checks: {column: [(start, end), ...]} periods flagged, when rows do not show them (e.g. gaps)
    intervals: dict = None

    def __init__(self, data = None, code=StatusCode.DEFAULT, intervals=None):
        self.data = data
        self.status_code = code
        self.intervals = intervals