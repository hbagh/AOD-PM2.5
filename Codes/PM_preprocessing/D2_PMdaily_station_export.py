# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 09:57:25 2020
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Join all PM-AOD and meteorological data-frames collected in different years
Phyton3 (env: base)
"""
# parameters

ini = 2013
end = 2019
ref_year = 2000

import pandas as pd
import os
from datetime import timedelta, date
import datetime


root_path = "G:\project\AQ\Data"


namePM = "PM.csv"
pathPM = os.path.join(root_path,"Ground_measurements", namePM)



dfPM = pd.read_csv(pathPM)






def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)
      
for year in range(ini, end+1):
    start_date = date(year, 1, 1)
    end_date = date(year+1, 1, 1)
           
    for single_date in daterange(start_date, end_date):
        # print("date: ", single_date)
        dfPMd = dfPM.loc[dfPM['date'] == str(single_date)]
        dfPMd.rename(columns={'PM 2.5 ug/m3': 'PM'}, inplace=True)
        dfPMd.drop(columns=['Station','station','date'], inplace=True)
        dfPMd = dfPMd[["lat", "long", "PM"]]
        dfPMd.dropna(inplace=True)
        if int(datestdtojd(str(single_date))) < 10:
                sday = "00"+ str(int(datestdtojd(str(single_date))))
        elif int(datestdtojd(str(single_date))) < 100 and int(datestdtojd(str(single_date))) > 9:
                sday = "0"+ str(int(datestdtojd(str(single_date))))
        else:
            sday = str(int(datestdtojd(str(single_date))))

        name_out = 'PM_station_' + str(year) + sday + '.csv'
        out_path = os.path.join(root_path, "Ground_measurements\daily_export", name_out)
        dfPMd.to_csv(out_path , index=False)
        print(name_out)

