"""
This DAG has the goal to create the SQLite Database and the tables to store
the devices to process data
"""
from datetime import timedelta
from scdata.io.device_api import ScApiDevice
from scdata._config import config
from scdata.utils.date import localise_date

# Set verbose level
config._out_level = 'DEBUG'

from airflow import DAG
from airflow.models import Variable
from airflow.hooks.sqlite_hook import SqliteHook
from airflow.operators.sqlite_operator import SqliteOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import urllib.request

DEVICES_URL = Variable.get("devices_url") #https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/master/airflow/devices/devices.txt
DEVICES_TABLE = Variable.get("post_devices_table") #devices

default_args = {
                "start_date": days_ago(1),
                "retries": 2,
                "retry_delay": timedelta(minutes=1),
                'provide_context': True
                }

dag = DAG(
            dag_id='devices_update',
            description = "Create and/or update tables to process devices",
            default_args = default_args,
            schedule_interval = "@daily"
          )


def get_devices_to_process(**kwargs):
    devices_raw = dict()
    for line in urllib.request.urlopen(kwargs['url']):
        devices_raw[line.decode('utf-8')] = {'last_reading': None, 'latest_postprocessing': None}
    return devices_raw

def get_devices_dates(**kwargs):
    ti = kwargs['ti']
    devices_raw = ti.xcom_pull(task_ids='get_devices_to_process')
    devices = devices_raw.copy()

    for device in devices:
        apidevice = ScApiDevice(device)
        location = apidevice.get_device_location()
        last_reading = apidevice.get_device_last_reading()
        
        try:
            latest_postprocessing = apidevice.get_postprocessing_info()['latest_postprocessing']
        except:
            latest_postprocessing = None
            pass

        devices[device]["last_reading"] = localise_date(last_reading, location).__str__()

        devices[device]["latest_postprocessing"] = localise_date(latest_postprocessing, location).__str__()

    return devices

def fill_devices_table(**kwargs):
    ti = kwargs['ti']
    devices = ti.xcom_pull(task_ids='get_devices_dates')
    conn_host = SqliteHook(sqlite_conn_id='sqlite_devices').get_conn()
    
    for device in devices:
        sql_insert = f"""INSERT OR REPLACE INTO {DEVICES_TABLE} 
                     (device, last_reading, latest_postprocessing)
                     VALUES ({device},
                             '{devices[device]["last_reading"]}',
                             '{devices[device]["latest_postprocessing"]}'
                            )
                     ;"""
        
        conn_host.execute(sql_insert)
        conn_host.commit()
    conn_host.close()

with dag:

    get_devices = PythonOperator(
        task_id = 'get_devices_to_process',
        python_callable = get_devices_to_process,
        op_kwargs = {'url': DEVICES_URL}
    )

    get_devices_dates = PythonOperator(
        task_id = 'get_devices_dates',
        python_callable = get_devices_dates    
    )

    create_devices_table = SqliteOperator(
        task_id = "create_devices_table",
        sql=f"""
                CREATE TABLE IF NOT EXISTS {DEVICES_TABLE}(
                device INT,
                last_reading DATE,
                latest_postprocessing DATE
                );
            """,
        sqlite_conn_id = "sqlite_devices"
        )

    fill_devices_table = PythonOperator(
        task_id = 'fill_devices_table',
        python_callable = fill_devices_table
    )


    # Create an unique index on the column device of
    # DEVICES_TABLE table. When inserting a new device,
    # if a prediction with a same device already exists,
    # it updates that particular row with the new values,
    # otherwise it inserts a new record.
    create_device_index = SqliteOperator(
        task_id="create_device_index",
        sql=f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_device
                ON {DEVICES_TABLE} (device)
                ;
            """,
        sqlite_conn_id="sqlite_devices"
        )

    create_devices_table >> create_device_index
    create_device_index >> get_devices
    get_devices >> get_devices_dates
    get_devices_dates >> fill_devices_table