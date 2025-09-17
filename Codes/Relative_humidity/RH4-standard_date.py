# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:20:43 2021


@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com

Description: 
Determination of standard date for in-situ measurements
"""
# Parameters

ini = 2013
end = 2019


import os
import pandas as pd
from datetime import datetime



def j2sdate(day_num, year):

    # adjusting day num
    day_num.rjust(3 + len(day_num), '0')
  
    # converting to date
    res = datetime.strptime(year + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
    return res
    

directory = 'C:\Project\AQ0\Data\Ground_measurements\synoptic_data'

for year in range (ini ,end+1):
    file_daily = "daily_ground" + str(year) + ".csv"
    path= os.path.join(directory, file_daily)
    df = pd.read_csv(path)
    df["Sdate"] = [j2sdate(str(int(day_num)), str(year)) for day_num in df['DOY']]

    fileOut_daily = "daily_ground" + str(year) + ".csv"
    path_daily= os.path.join(directory, fileOut_daily)
    df.to_csv(path, index = False)
    print("processed data for " + str(year))