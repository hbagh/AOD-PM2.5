# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:26:52 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Monty PM calculation
env: Python3
"""
City = "Tehran"
ini = 2013
end = 2019
_FillValue = -9999

import glob
import datetime
import numpy as np
import os
import pandas as pd
root = 'G:\project\AQ\data-Iran_Tehran'


def jdtodatestd (jdate):
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)
mask_path = os.path.join(root, "Tehran_mask.npz")
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]

top_left_lat = 35.88305
top_left_lon = 51.0890
bottom_right_lat = 35.5632
bottom_right_lon = 51.6084 
# 51.0725, 51.6233, 35.5873, 35.8318   
grid_lat = np.linspace( top_left_lat, bottom_right_lat, mask.shape[0])
grid_lon = np.linspace(top_left_lon, bottom_right_lon, mask.shape[1])
    
long_data , lat_data = np.meshgrid(grid_lon, grid_lat)

List = []

path = os.path.join(root, 'Grid\PM\daily_final_int_PM')
for file in glob.glob(path + '\*.npz'):
    date=jdtodatestd(file[-11:-4])
    List.append((date.year, date.month, file))


per = []
for year in range (ini, end+1):
    yList = [t[0:3] for t in List if t[0] == year]
    print ("Year:", year)
    for month in range (1,13):
        mlist = [t[0:3] for t in yList if t[1] == month]
        print(len(mlist))
        pm_list = []
        for i in range(len(mlist)):
            loaded_pm = np.load(mlist[i][2])
            pm = loaded_pm["pm"]
            pm[pm == _FillValue] = np.nan
            pm_list.append(pm)
        pmm  = np.nanmean(pm_list, axis=0)
        
        percent = np.round(np.sum(np.isnan(pmm)/(pmm.shape[0]*pmm.shape[1])),2)
        per.append(percent)
        print("Year:{}, Month:{}, Gap percent: {}".format(year, month, percent))
        pmm[np.isnan(pmm)]= -9999
        if month < 10: 
            smonth = "0"+str(month)
        else:
            smonth = str(month) 
        if not os.path.exists(os.path.join(root, 'Grid\PM\monthly_final_int_PM')):
            os.mkdir(os.path.join(root, 'Grid\PM\monthly_final_int_PM'))
        output_name = "montly_PM_"+smonth+"_"+str(year)
        output = os.path.join(root, 'Grid\PM\monthly_final_int_PM', output_name)
        np.savez_compressed(output, pm=pmm)
        print(output_name+" was processed and generated")
        # Xd = []

        # for r in range(mask.shape[0]):
        #     for c in range (mask.shape[1]):
        #         if pmm[r,c] == _FillValue:
        #             continue
        #         else:
        #             Xd.append(np.asarray([lat_data[r,c], long_data[r,c], pmm[r,c]]))
                       
        # Xd = np.asarray(Xd)
        # if Xd.size == 0 :
        #     continue

        # df = pd.DataFrame(Xd, columns = ['lat', 'long', 'PM'])

        # out_path = os.path.join(root, 'Grid\PM\monthly_PM')
    
        # if not os.path.exists(out_path):
        #     os.mkdir(out_path)
        
        # file_name = "monthly_PM_est_" + smonth + "_"+str(year) + '.csv'
        # df.to_csv(os.path.join(out_path, file_name) ,index = False)
        






        







            
        
    

    


