from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TestOptions(BaseModel):
    cache: Optional[bool] = False

class Metric(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = ''
    module: Optional[str] = "scdata.device.process"
    function: str
    unit: Optional[str] = ''
    post: Optional[bool] = False
    args: Optional[dict] = None
    kwargs: Optional[dict] = None

class Sensor(BaseModel):
    id: int
    name: str
    description: str
    unit: Optional[str] = None

class Source(BaseModel):
    type: str = 'api'
    module: str = 'smartcitizen_connector'
    handler: str = 'SCDevice'

class APIParams(BaseModel):
    id: int

class FileParams(BaseModel):
    id: int # To be compatible with API id
    path: str

class CSVParams(FileParams):
    header_skip: Optional[List[int]] = [1,2,3]
    index: Optional[str] = 'TIME'
    separator: Optional[str] = ','
    tzaware: Optional[bool] = True
    timezone: Optional[str] = "UTC"

class DeviceOptions(BaseModel):
    clean_na: Optional[bool] = None
    frequency: Optional[str] = '1Min'
    resample: Optional[bool] = False
    min_date: Optional[str] = None
    max_date: Optional[str] = None

class Blueprint(BaseModel):
    meta: dict = dict()
    metrics: List[Metric] = []

class Name(BaseModel):
    id: int
    name: str
    description: str
    unit: str
