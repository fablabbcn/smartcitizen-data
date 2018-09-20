import math
import pandas as pd

def assignSession(dataframe):
    old_col = dataframe.index.hour
    bins = [-1,5,11,17,23]
    new_lab = ['Night','Morning','Afternoon','Evening']
    new_col = pd.cut(old_col, bins, labels=new_lab)
    
    new_df = dataframe.assign(session = new_col)
    return new_df

    
def groupbyfreq(dataframe, freqq, lis_pollu, calc): 
    """
        dataframe - type > panadas dataframe 
        freqq - panadas freq tags | type > string 
                ex. if you want to group datetime index by a 2 hour frequency, then use '2H' 
                see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases 
        lis_pollu - subset of of coulmn names from original dataframe | type > list of strings
                    using ['CO_AD_BASE', 'NO2_AD_BASE', 'O3_AD_BASE', 'EXT_PM_1', 'EXT_PM_25', 'EXT_PM_10', 'session']
                    but can add or take away some column names 
        calc - calculating average or sum of groups | type > int 
               0 to calculate sum
               1 to calculate average
    """
    df = pd.DataFrame()
    if calc:
        df = dataframe.resample(freqq, label="right").mean()
    else:
        df = dataframe.resample(freqq, label="right").sum()
    
    dfgrouped = assignSession(df[lis_pollu])
    #display(dfgrouped)
    
    return dfgrouped

def avgbysess(dataframe, lis_pollu):
    df = dataframe.groupby(['session']).mean()
    
    df = df.abs()
    new_df = df[lis_pollu]
    
    new_df['total_poll'] = new_df.sum(axis=1)
    new_df = new_df.sort_values(by='total_poll', ascending=0)
    new_df = new_df.drop('total_poll', 1)

   
    #display(new_df)
    return new_df

def compAvg(dataframe, freqq, lis_pollu, calc):
    avg_sessdf = avgbysess(dataframe,lis_pollu)
    fulldf = groupbyfreq(dataframe, freqq, lis_pollu, calc)
    
    indexs = fulldf.index
    columns = ['CO_AD_BASE_60', 'NO2_AD_BASE_60', 'O3_AD_BASE_60', 'EXT_PM_1', 'EXT_PM_25', 'EXT_PM_10', 'session']
    vis_df = pd.DataFrame(index=indexs, columns=columns, dtype='float')
    vis_df = vis_df.fillna(0)
    vis_df['session'] = vis_df['session'].astype(str) 
    
    for index, row in fulldf.iterrows():
        cell_ses = row['session']
        #print index.hour, cell_ses
        vis_df.at[index, 'session'] = cell_ses
    
        for column_name, column_series in dataframe.iteritems():
            if column_name not in lis_pollu :
                continue
            elif column_name == "session":
                continue
            else:
                #print cell_ses, column_name
                day_avg = row[column_name]
                ses_avg = avg_sessdf.loc[cell_ses, column_name]

                if math.isnan(day_avg) or math.isnan(ses_avg):
                    vis_df.at[index, column_name] = 0
                else:
                    vis_df.at[index, column_name] = day_avg
        
    return vis_df