# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:35:24 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Estimate PM2.5 using a ML model e.g. XGBoost
env: Python3
"""
Date = 2018056

# 25 feb 2018
varAOD = 'nAODm'
root = 'G:\project\AQ0\Data\Grid'
root_path = 'G:\project\AQ0\Data'
folderAOD = "AOD"
foldermet = "MET"
_FillValue = -28672

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import datetime
import pandas as pd

def jdtodatestd (jdate):
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)


    
# lat , long
mask_path = os.path.join("G:\project\AQ0\Data", "Tehran_mask.npz")
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

    # Model 
    # load model from file
from xgboost import XGBRegressor
import xgboost as xgb
path_model = os.path.join('G:\project\AQ0\Data\model_xgboost', "PMXGB")
    
reg = XGBRegressor()
booster = xgb.Booster()
booster.load_model(path_model)
reg._Booster = booster


data = []    
# AOD
folder = "daily"
filename = "AOD" + "_" + str(Date) + ".npz"
path = os.path.join(root, folderAOD, folder, filename)
loaded_aod = np.load(path)
data.append(loaded_aod["ac"])
    

data.append(lat_data)
data.append(long_data)
    
    # Qbest
folder = "daily\Qb"
filename =  "Qb" + "_" + str(Date) + ".npz"
path = os.path.join(root, folderAOD, folder, filename)
loaded_Qb = np.load(path)
data.append(loaded_Qb["qp"])
    
    # met data
      
LIST = ['d2m', 't2m', 'blh', 'sp', 'lai_hv', 'ws10', 'wd10', 'uvb']
    
for var in LIST:
    folder = 'daily'
    subfolder = "daily_" + var
    filename = "daily_" + var + "_" + str(Date) + ".npz"
    path = os.path.join(root, foldermet, folder, subfolder, filename)
    loaded_met = np.load(path)
    data.append(loaded_met["met"])

    
    
# data normalization
    
name = 'train' + ".csv"
path = os.path.join(root_path, name)
df = pd.read_csv(path)
    
X = df[[varAOD, 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
        'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'RH', 'DOY']]
    

    
scaler = MinMaxScaler()
Xs = scaler.fit_transform(X)
    
def get_AOD(x, y, data, Qbest,_FillValue, grid_size):
    th1=int((grid_size-1)*0.5)
    th2 = int(th1+1)
    if x < th1:
        x+=1
    if x > data.shape[0]-th2:
        x-=2
    if y < th1:
        y+=1
    if y > data.shape[1]-th2:
        y-=2
    x_by_x = data[x-th1:x+th2, y-th1:y+th2]
    qbest = Qbest[x-th1:x+th2, y-th1:y+th2]
    # qmed = Qmed[x-th1:x+th2, y-th1:y+th2]
    x_by_x=x_by_x.astype(float)
    mask = x_by_x==float(_FillValue)
    x_by_x[mask]=np.nan
    qbest[mask]=np.nan 
    # qmed[mask]=np.nan 
    Num = np.count_nonzero(~np.isnan(x_by_x))
    qbestm = np.nanmean(qbest)
    # qmedm = np.nanmean(qmed)
    aodm = np.nanmean(x_by_x)
    std=np.nanstd(x_by_x)
    return aodm, qbestm, std, Num

    
def RH(d2m, t2m):
    return ((d2m-0.1*t2m +112)/(112+0.9*t2m))**(1/0.125)
    

Xd = []
pm = np.zeros((mask.shape[0], mask.shape[1]))
for r in range(mask.shape[0]):
    for c in range (mask.shape[1]):
        AODm, qbestm, std, num = get_AOD(r, c, data[0], data[3],_FillValue, 3)
            
        if AODm == _FillValue:
            pm[r,c] = -9999     
        elif num < 3:
            pm[r,c] = -9999
        elif std > 500:
            pm[r,c] = -9999
        else:
            lat = data[1][r,c]
            long = data[2][r,c]
            # Prob_bestm = data[3][r,c]
            d2m = data[4][r,c]
            t2m = data[5][r,c]
            blh = data[6][r,c]
            sp = data[7][r,c]
            lai_hv = data[8][r,c]
            ws10 = data[9][r,c]
            wd10 = data[10][r,c]
            uvb = data[11][r,c]
            nAODm = (AODm*0.001)/(blh/1000)
            rh = RH(d2m, t2m)
            DOY = int(str(Date)[4:9])
            fRH = (1-rh)**(-1)
            
            Xd.append(np.asarray([nAODm, lat, long, qbestm, d2m, t2m, blh, sp, lai_hv,
                     ws10, wd10, uvb, rh, DOY, r, c, fRH]))
                
                
Xd = np.asarray(Xd)
if Xd.size == 0:
    print('there is not such a file')
    
  
else:
    xs = scaler.fit_transform(Xd[:,0:14])
    y_pred = reg.predict(xs).reshape(-1,1)    
    df = np.concatenate((Xd, y_pred), axis = 1)
    df = pd.DataFrame(df, columns = ['AODn', 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
                                      'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'RH', 'DOY', 'r',
                                      'c', 'fRH', 'PM_pred'])
    def realPM(row):
        return row["PM_pred"]/row["fRH"]
    
    df['PM'] = df.apply(realPM, axis = 1)
    
    df_export  = df[["lat", "long", "PM", "r", "c"]]
    for i in range(df.shape[0]):
        row = int(df["r"].iloc[i])
        col = int(df["c"].iloc[i])
        PM = df["PM"].iloc[i]
        pm[row, col] = PM
        
    file_name = "PM_est_" + str(Date) + '.npz'
    out_path = os.path.join(root_path, 'Grid\PM\daily')
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    np.savez_compressed(os.path.join(out_path, file_name), pm=pm)
    print (os.path.join(out_path, file_name))
    
    file_name = "PM_est_" + str(Date) + '.csv'
    df_export.to_csv(os.path.join(out_path, file_name) ,index = False)

   


    
      

    












