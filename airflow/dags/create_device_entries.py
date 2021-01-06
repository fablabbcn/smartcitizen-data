'''
This DAG has the goal to create the Postgresql Database and the tables to store
the devices to process data
'''

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
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

DEVICES_TABLE = Variable.get("post_devices_table", default_var = "devices") #devices

default_args = {
                'start_date': days_ago(1),
                'retries': 2,
                'retry_delay': timedelta(minutes=1),
                'provide_context': True
                }

dag = DAG(
            'devices_update',
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

        last_reading = apidevice.get_device_last_reading()
        added_at = apidevice.get_device_added_at()

        # Check last reading
        if last_reading is None or last_reading == 'null': last_reading = added_at
        devices[device]['last_reading'] = localise_date(last_reading, 'UTC').__str__()

        #Check latest_postprocessing
        if devices[device]['latest_postprocessing'] is None or devices[device]['latest_postprocessing'] == 'null':
            devices[device]['latest_postprocessing'] = localise_date(added_at, 'UTC').__str__()

    return devices

def fill_devices_table(**kwargs):
    ti = kwargs['ti']
    devices = ti.xcom_pull(task_ids='get_devices_dates')
    conn_host = PostgresHook(postgres_conn_id='postgres_default').get_conn()

    for device in devices:
        sql_insert = f"""INSERT INTO {DEVICES_TABLE}
                     (device, last_reading, latest_postprocessing, updated_at, blueprint_url, hardware_url, failed)
                     VALUES ({device},
                             '{devices[device]["last_reading"]}',
                             '{devices[device]["latest_postprocessing"]}',
                             '{devices[device]["updated_at"]}',
                             '{devices[device]["blueprint_url"]}',
                             '{devices[device]["hardware_url"]}',
                             0
                            )
                     ON CONFLICT (device) DO UPDATE
                       SET last_reading = excluded.last_reading,
                           latest_postprocessing = excluded.latest_postprocessing,
                           updated_at = excluded.updated_at,
                           blueprint_url = excluded.blueprint_url,
                           hardware_url = excluded.hardware_url,
                           failed = excluded.failed;
                     ;"""

        with conn_host.cursor() as cur:
            cur.execute(sql_insert)
        conn_host.commit()
    conn_host.close()

get_devices = PythonOperator(
    task_id = 'get_devices_to_process',
    python_callable = get_devices_to_process,
    dag = dag
)

get_devices_dates = PythonOperator(
    task_id = 'get_devices_dates',
    python_callable = get_devices_dates,
    dag = dag
)

create_devices_table = PostgresOperator(
    task_id = "create_devices_table",
    sql=f"""
            CREATE TABLE IF NOT EXISTS {DEVICES_TABLE}(
            device INT,
            last_reading TIMESTAMP,
            latest_postprocessing TIMESTAMP,
            updated_at TIMESTAMP,
            blueprint_url TEXT,
            hardware_url TEXT,
            failed INT
            );
        """,
    postgres_conn_id = "postgres_default",
    dag = dag
    )

fill_devices_table = PythonOperator(
    task_id = 'fill_devices_table',
    python_callable = fill_devices_table,
    dag = dag
)

'''
Create an unique index on the column device of
DEVICES_TABLE table. When inserting a new device,
if a prediction with a same device already exists,
it updates that particular row with the new values,
otherwise it inserts a new record.
'''
create_device_index = PostgresOperator(
    task_id='create_device_index',
    sql=f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_device
            ON {DEVICES_TABLE} (device)
            ;
        """,
    postgres_conn_id='postgres_default',
    dag = dag
    )

create_devices_table >> create_device_index
create_device_index >> get_devices
get_devices >> get_devices_dates
get_devices_dates >> fill_devices_table