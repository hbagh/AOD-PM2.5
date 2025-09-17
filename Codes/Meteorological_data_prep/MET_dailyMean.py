# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:53:15 2020
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com

Description: 
Prepare meteorological data for interpolation
Daily average of variables

Python 3, (env: base)
"""
# Parameters

ini = 2019
end = 2019
name = "met"
# Reference year: Satrting date
ref = 2000

import os
import pandas as pd
import datetime
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def datestdtojd (stddate):
    fmt='%Y-%m-%d %H:%M:%S'
    sdtdate = datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)
def HOUR (stddate):
    fmt='%Y-%m-%d %H:%M:%S'
    sdtdate = datetime.strptime(stddate, fmt)
    h = sdtdate.hour
    return(h)

directory = 'G:\project\AQ\data-Iran_Tehran\MET'
fileOut_daily = "daily_" + name + ".csv"
fileOut_h = "half_" + name + ".csv"



dfm0 = []
dfmh0 = []

for year in range (ini ,end+1):
    csv_file = str(year) + "_" + name + ".csv"
    path= os.path.join(directory, csv_file)
    
    df = pd.read_csv(path)


    # df["Sdate"] = [d.split()[0] for d in df["time"]]
    df['Sdate'] = pd.to_datetime(df['time'], errors='coerce')
    df["date"] = [d.date() for d in df["Sdate"]]
    df['Hour'] = [HOUR(date) for date in df['time']]
    df["DOY"] = [datestdtojd(date) for date in df["time"]]
    for day in df.date.unique():
        df_day = df.loc[df['date'] == day]
        print("Day:", str(day), "Year:" , year)
        for lon in df.longitude.unique():
            df_lon = df_day.loc[df_day['longitude'] == lon]
            for lat in df.latitude.unique():
                df_lat = df_lon.loc[df_lon['latitude'] == lat]
                df_h = df_lat.loc[(df_lat['Hour'] >= 4) & (df_lat['Hour'] <= 14)]
                dfm_day = df_lat.mean()
                dfm_day["Year"] = year
                dfm_day["Sdate"] = day
                dfm0.append(dfm_day)
                dfm_h = df_h.mean()
                dfm_h["Year"] = year
                dfm_h["Sdate"] = day
                dfmh0.append(dfm_h)
    dfavg = dfm0[0]
    dfhavg = dfmh0[0]
    for i in range(1, len(dfm0)):
        dfavg = pd.concat([dfavg, dfm0[i]], axis=1)
        dfhavg = pd.concat([dfhavg, dfmh0[i]], axis=1)
    dfavg = dfavg.T
    dfavg.reset_index(drop=True, inplace=True)
    dfhavg = dfhavg.T
    dfhavg.reset_index(drop=True, inplace=True) 
             
    
    fileOut_daily = "daily_"+str(year-ref)+name + ".csv"
    fileOut_h = "half_"+str(year-ref)+name + ".csv"
    path_daily= os.path.join(directory, fileOut_daily)
    path_h= os.path.join(directory, fileOut_h)
    dfavg.to_csv(path_daily)
    dfhavg.to_csv(path_h)
    print("processed data for " + str(year))


