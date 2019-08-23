import pandas as pd
import os
import csv


# def get_sample_data():
#     output_path = '/home/dataportal/data_samples/clean_monitoring_stations.json'

#     sensor_ids = ['1340424','5801123']
#     lat = [41.387784, 41.441131]
#     lon = [2.114951, 2.191972]

#     df_full = pd.DataFrame()

#     i=0
#     for sensor_id in sensor_ids:
#         url = 'https://data.waag.org/apimuv/getSensorData?sensor_id=' + sensor_id
#         df = pd.read_json(url)
#         df = df.dropna(axis=1, how='all')
#         df_full = pd.concat([df_full, df], sort=False)
#         df_full['lat'] = lat[i]
#         df_full['lon'] = lon[i]    
#         i+=1

#     df_full = df_full.reset_index(drop=True)

#     print(df_full.head())
#     df_full.to_json(output_path, orient='records')

testNames = ['dB', 'h', 'no2op1', 'no2op2', 'o3op1', 'o3op2', 'p10', 'p25', 't', 'time']
targetNames= ['NOISE_A', 'HUM', 'GB_2W', 'GB_2A', 'GB_3W', 'GB_3A', 'EXT_PM_10', 'EXT_PM_25', 'TEMP', 'Time']

def get_bcn_sensors_csv():
    bcn_sensor_ids =  ['1340424','1340511','5801167','5799871','5801118',
                        '5800177','5801988','5802283','1340722','5801123','5800147','5800390']

    df_full = pd.DataFrame()
    for sensor_id in bcn_sensor_ids:
        print ('Requesting device {} ({}/{})'.format(sensor_id, bcn_sensor_ids.index(sensor_id)+1, len(bcn_sensor_ids)))
        url = 'https://data.waag.org/api/muv/getSensorData?sensor_id=' + sensor_id
        df = pd.read_json(url)
        for i in range(len(targetNames)):
            if not (testNames[i] == '') and not (testNames[i] == targetNames[i]) and testNames[i] in df.columns:
                df.rename(columns={testNames[i]: targetNames[i]}, inplace=True)
                # print('Renaming column *{}* to *{}*'.format(testNames[i], targetNames[i])))
        # df_full = pd.concat([df_full, df])

        df.drop('id', axis=1, inplace=True)
        df = df.set_index('Time')
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Europe/Madrid')
        # df.index = pd.to_datetime(df.index).tz_convert('Europe/Madrid')
        min_date = '2019-07-02'
        df = df[df.index > min_date]
        if len(df.index) > 0:
            print ('Device {} contains {} points'.format(sensor_id, len(df.index)))
            print ('First measurement on {} - last measurement on {}'.format(df.index[0], df.index[-1]))
            print ('------')
            df.to_csv("MUV_{}.csv".format(sensor_id), index=True, quoting=csv.QUOTE_NONNUMERIC)
        else:
            print ('No data between dates')
    # print(df.head())

if __name__ == "__main__":
    # get_sample_data()
    get_bcn_sensors_csv()