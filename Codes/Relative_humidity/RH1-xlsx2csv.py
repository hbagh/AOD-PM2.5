# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:41:16 2021

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Convert tri-hourly relative humidity measurments from ground stations to csv 
(Excel (xls) iputs)

"""
# Parameters



root = "C:\Project\AQ0\Data"
data_path = "Ground_measurements\synoptic_data"



# libraries
import os
import pandas as pd
import glob
import numpy as np


# file path
file_path = glob.glob(os.path.join(root, data_path) +'\*.xlsx')
 
station = pd.DataFrame(pd.read_excel(os.path.join(root, 'met_station_location.xlsx')))


for path in file_path:
    print(path)

    df = pd.DataFrame(pd.read_excel(path)) 

    df['new_date'] = [str.split(date)[0] for date in df['date']]
    df["latitude"] = np.nan
    df["longitude"] = np.nan

    
    for i in range(len(df['station_id'])):
        if   df['station_id'][i] == 40777:
            idx = 0
        elif df['station_id'][i] == 40754:
            idx = 1
        elif df['station_id'][i] == 40755:
            idx = 2
        elif df['station_id'][i] == 99320:
            idx = 3
        elif df['station_id'][i] == 99369:
            idx = 4
        elif df['station_id'][i] == 40756:
            idx = 5
        elif df['station_id'][i] == 40751:
            idx = 6
        elif df['station_id'][i] == 99370:
            idx = 7
        elif df['station_id'][i] == 99331:
            idx = 8
        elif df['station_id'][i] == 99375:
            idx = 9
        elif df['station_id'][i] == 99406:
            idx = 10
        elif df['station_id'][i] == 99366:
            idx = 11
        df['latitude'][i]= station['lat'][idx]
        df['longitude'][i]= station['long'][idx]


    out_path =  path[:-17]+'.csv'
    
    df.to_csv(out_path, index=False)


print("\n All xls were converted")










