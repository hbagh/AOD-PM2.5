# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:29:42 2021
Interpolate (climate) data at a grid format
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
env: krig
"""
import pandas as pd
import os
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import numpy as np
from datetime import timedelta, date
from sklearn import preprocessing

# parameters
ini = 2013
end = 2019

met_name = "met"     # for 'd2m', 't2m', 'blh', 'lai_hv', 'lai_lv', 'sp', 'tp', 'ws10', 'wd10'
    # for cdir, uvb

#LIST = ['d2m', 't2m', 'blh', 'lai_hv', 'lai_lv', 'sp', 'tp', 'ws10', 'wd10', cdir, uvb]
LIST = ['cdir']
MV = ["power", "universal"]

ref = 2000

root_path = "G:\project\AQ1\Data"
mask_path = os.path.join(root_path, "Tehran_mask.npz")  


top_left_lat = 35.8305 
top_left_lon = 51.0890
bottom_right_lat = 35.5632
bottom_right_lon = 51.6084   
  



def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
    

def krig2D(data, grid_lon, grid_lat, method, variogram):
    scaler = preprocessing.MinMaxScaler()
    target_scaler = preprocessing.MinMaxScaler()
  
    

    lon = scaler.fit_transform(data[:, 0].reshape(-1,1))
    lat = scaler.fit_transform(data[:, 1].reshape(-1,1))
    val = target_scaler.fit_transform(data[:,2].reshape(-1,1))
    
    grid_lon_scaled = scaler.fit_transform(grid_lon.reshape(-1,1))
    grid_lat_sclaed = scaler.fit_transform(grid_lat.reshape(-1,1))
    if method == 'universal':
        UK = UniversalKriging(lon, lat, val,
                              variogram_model=variogram,
                              pseudo_inv_type="pinvh")
        k,_ = UK.execute("grid", grid_lon_scaled, grid_lat_sclaed)
        
    elif method == 'ordinary':
        OK = OrdinaryKriging(lon, lat, val,
                             variogram_model=variogram,
                             pseudo_inv_type="pinvh")
        k,_ = OK.execute("grid", grid_lon_scaled, grid_lat_sclaed)
    return target_scaler.inverse_transform(k)

             

mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]         
            
grid_lat = np.linspace( top_left_lat, bottom_right_lat, mask.shape[0])
grid_lon = np.linspace(top_left_lon, bottom_right_lon, mask.shape[1])

method = MV[1]
variogram = MV[0]


for year in range(2013, end+1):

    pathMETd = os.path.join(root_path, "MET", "daily_"+str(year-ref)+met_name+".csv")  
    df1 = pd.read_csv(pathMETd)
    df1 = df1.sort_values(["Sdate"], axis=0, ascending=[True])


    
    for var in LIST:
        print(var)
        start_date = date(year, 1, 1)
        end_date = date(year+1, 1, 1)
        for single_date in daterange(start_date, end_date):
            print("date: ", single_date)
            sub_df1 = df1.loc[df1['Sdate'] == str(single_date)]
            
            
    
            data=sub_df1[["latitude", "longitude", var]].values

            var_grid = krig2D(data, grid_lon, grid_lat, method, variogram)
            folder = "daily_"+var
            out_path = os.path.join(root_path, "Grid\MET", folder)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            if int(sub_df1["DOY"].iloc[0]) < 10:
                sday = "00"+ str(int(sub_df1["DOY"].iloc[0]))
            elif int(sub_df1["DOY"].iloc[0]) < 100 and int(sub_df1["DOY"].iloc[0]) > 9:
                sday = "0"+ str(int(sub_df1["DOY"].iloc[0]))
            else:
                sday = str(int(sub_df1["DOY"].iloc[0]))
                
            out_name = "daily_"+ var + "_" + str(int(sub_df1["Year"].iloc[0])) + sday
            outputv2 = os.path.join(out_path, out_name)
            np.savez_compressed(outputv2, met=var_grid)
            








