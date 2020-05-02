# I2CAT RESEARCH CENTER - BARCELONA - MARC ROIG (marcroig@i2cat.net)
import io
import os
import requests
import csv
import pandas as pd
import matplotlib.pyplot as plt


def build_request(station_id, start_date, end_date, contaminants):
    contaminant_filter="contaminant in " + str(contaminants)
    date_filter = "data between " + start_date + " and " + end_date
    #sql_filter = "&$where=" + date_filter # + " AND " + contaminant_filter
    sql_filter = "&$where=" + date_filter
    station_id_filter = "codi_eoi=" + station_id
    
    query = "https://analisi.transparenciacatalunya.cat/resource/" \
            "uy6k-2s8r.csv?" + station_id_filter + sql_filter

    print("\nEXECUTING QUERY: " + query)
    return query


def api_request(query):
    s=requests.get(query).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    print("\nSUCCESSFUL REQUEST, NUMBER OF RECORDS: " + str(df.shape[0]))
    return df


def filter_columns(df):
    measures = ['h0' + str(i) for i in range(1,10)]
    measures = measures + ['h' + str(i) for i in range(10,25)]
    validations = ['v0' + str(i) for i in range(1,10)]
    validations  = validations + ['v' + str(i) for i in range(10,25)]
    new_measures_names = list(range(1,25))

    columns = ['contaminant', 'data'] + measures# + validations

    df_subset = df[columns]
    df_subset.columns  = ['contaminant', 'date'] + new_measures_names
    print("\nSUCCESSFUL COLUMN FILTERING")
    return df_subset


def pivot_data(df_subset, contaminants):
    print("\nSTART DATA PIVOTING")
    df_pivot = pd.DataFrame([])

    for contaminant in contaminants:
        
        df_temp= df_subset.loc[df['contaminant']==contaminant].drop('contaminant', 1).set_index('date').unstack().reset_index()
        df_temp.columns = ['hours', 'date', contaminant]
        df_temp['date'] = pd.to_datetime(df_temp['date'])

        """
        # DEPRECATED
        for index in df_temp.index:
            df_temp['date'][index] = df_temp.copy['date'][index] +  pd.DateOffset(hours=int(df_temp.copy()['hours'][index]))
        """
        timestamp_lambda = lambda x: x['date'] + pd.DateOffset(hours=int(x['hours']))

        df_temp['date'] = df_temp.apply( timestamp_lambda, axis=1)
        df_temp = df_temp.set_index('date')
        df_pivot[contaminant] = df_temp[contaminant]
        
    df_pivot = df_pivot.sort_index()
    print("\nDATA PIVOTING FINISHED")
    print("\nNUMBER OF RECORDS AFTER PIVOTING: " + str(df_pivot.shape[0]))
    print("\n")
    print(df_pivot.head(3))
    return df_pivot


def temporal_plot(df_pivot):
    df_pivot.plot(figsize=(15,5))
    plt.show()


"""
0. Define parameters you want to query.
1. Build query from parameters 
2. Get raw data from api
3. Get interesting columns, modify column names
4. Pivot data: 
    raw data ==> rows are contaminants per day, columns are the measures for 24h
    pivoted data ==> rows have an unique timestamp, columns are all the contaminats for the timestamp
5. Save results
6. (OPTIONAL) plot the data over time
"""

# PARAMETERS
output_file = 'csic_data.csv'
output_path = os.path.join(os.getcwd(),output_file) # WARNING: MODIFY IT!

station_name = "Barcelona%20(Palau%20Reial)"
station_id="8019057"

start_date = "'2019-05-23T00:00:00'"
end_date = "'2019-06-30T00:00:00'"

contaminants =  ('NO2', 'PM10', 'CO', 'NO', 'NOX') # ('SO2', 'NO', 'NO2', 'NOX', 'CO', 'PM10')

# STEPS 1-6
if __name__ == "__main__":

    query = build_request(station_id, start_date, end_date, contaminants)

    df = api_request(query)

    df_subset = filter_columns(df)

    df_pivot = pivot_data(df_subset, contaminants)

    df_pivot.to_csv(output_path, quoting = csv.QUOTE_NONNUMERIC)

    # temporal_plot(df_pivot)

    print("\nCSV FILE AT: " + output_path)
    
    
