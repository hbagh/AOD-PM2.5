# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:33:49 2021

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com

Description: 
Prepare in_situ data for interpolation
Daily average of in_situ measurements
"""
# Parameters

ini = 2013
end = 2019


import os
import pandas as pd
import datetime
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def datestdtojd (stddate):
    if len(stddate) <= 10:
        fmt='%d/%m/%Y'
    if len(stddate) > 10:
        stddate = str.split(stddate)[0]
        fmt='%m/%d/%Y'

    sdtdate = datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(int(jdate))


directory = 'C:\Project\AQ0\Data\Ground_measurements\synoptic_data'





dfm0 = []

#df['Sdate'] = pd.to_datetime(df['new_date'], errors='coerce')

for year in range (ini ,end+1):
    path= os.path.join(directory, str(year)+'.csv')
    df = pd.read_csv(path)
    df["DOY"] = [datestdtojd(date) for date in df['date']]
    df.drop(columns = ['station_name', 'region_id', 'region_name', 'date',
       'data', 'dd', 'ff', 't', 'td', 'p', 'p0', 'geops', 'tmin', 'tmax',
       'radglo24', 'rrr24', 'twet', 'ff_max', 'dd_max', 'ff_gust', 'dd_gust',
       'tsoil', 'ew', 'ews', 'pqnh', 'new_date', 'year'], inplace = True)

    for day in df['DOY'].unique():
        df_day = df.loc[df['DOY'] == day]
        print("Day:", str(day), "Year:" , year)
        for idx in df['station_id'].unique():
            df_idx = df_day.loc[df_day['station_id'] == idx]
            dfm_day = df_idx.mean()
            dfm_day['station_id']= idx
            dfm_day["Year"] = year
            dfm_day["DOY"] = day
            dfm0.append(dfm_day)

    dfavg = dfm0[0]
    for i in range(1, len(dfm0)):
        dfavg = pd.concat([dfavg, dfm0[i]], axis=1)

    dfavg = dfavg.T
    dfavg.reset_index(drop=True, inplace=True)
    dfavg.dropna(inplace = True)

             
    
    fileOut_daily = "daily_ground" + str(year) + ".csv"
    path_daily= os.path.join(directory, fileOut_daily)
    dfavg.to_csv(path_daily, index = False)
    print("processed data for " + str(year))