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
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import urllib.request

DEVICES_TABLE = Variable.get('post_devices_table') #post_devices_table

default_args = {
                'start_date': days_ago(1),
                'retries': 2,
                'retry_delay': timedelta(minutes=1),
                'provide_context': True
                }

dag = DAG(
            dag_id='devices_update',
            description = 'Create and/or update tables to process devices',
            default_args = default_args,
            schedule_interval = '@daily'
          )


def get_devices_to_process(**kwargs):
    df = ScApiDevice.search_query(key = 'postprocessing_info', value = 'not_null', full = True)
    devices_raw = dict()
    for item in df.index:
        devices_raw[item] = df.loc[item, 'postprocessing_info']
    return devices_raw

def get_devices_dates(**kwargs):
    ti = kwargs['ti']
    devices_raw = ti.xcom_pull(task_ids='get_devices_to_process')
    devices = devices_raw.copy()

    for device in devices:
        apidevice = ScApiDevice(device)
        location = apidevice.get_device_location()
        last_reading = apidevice.get_device_last_reading()
        
        devices[device]['last_reading'] = localise_date(last_reading, location).__str__()

    return devices

def fill_devices_table(**kwargs):
    ti = kwargs['ti']
    devices = ti.xcom_pull(task_ids='get_devices_dates')
    conn_host = PostgresHook(postgres_conn_id='postgres_default').get_conn()
    
    for device in devices:
        sql_insert = f"""INSERT OR REPLACE INTO {DEVICES_TABLE} 
                     (device, last_reading, latest_postprocessing, updated_at, blueprint_url, hardware_url)
                     VALUES ({device},
                             '{devices[device]["last_reading"]}',
                             '{devices[device]["latest_postprocessing"]}',
                             '{devices[device]["updated_at"]}',
                             '{devices[device]["blueprint_url"]}',
                             '{devices[device]["hardware_url"]}'
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

    create_devices_table = PostgresOperator(
        task_id = "create_devices_table",
        sql=f"""
                CREATE TABLE IF NOT EXISTS {DEVICES_TABLE}(
                device INT,
                last_reading DATE,
                latest_postprocessing DATE,
                updated_at DATE,
                blueprint_url TEXT,
                hardware_url TEXT
                );
            """,
        postgres_conn_id = "postgres_default"
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
    create_device_index = PostgresOperator(
        task_id='create_device_index',
        sql=f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_device
                ON {DEVICES_TABLE} (device)
                ;
            """,
        postgres_conn_id='postgres_default'
        )

    create_devices_table >> create_device_index
    create_device_index >> get_devices
    get_devices >> get_devices_dates
    get_devices_dates >> fill_devices_table