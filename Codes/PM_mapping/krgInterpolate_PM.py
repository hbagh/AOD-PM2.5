# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:39:32 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Daily PM Interpolation: Kriging
env: krig
"""

root = 'G:\project\AQ\data-Iran_Tehran'

import os
from sklearn.model_selection import GridSearchCV
from pykrige.rk import Krige
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import glob
import pandas as pd
import numpy as np

     
def krig2D(x,y,z, method, variogram):
   
    if method == 'universal':
        interpolate = UniversalKriging(x, y, z,
                              variogram_model=variogram,
                              pseudo_inv_type="pinvh")

    
    elif method == 'ordinary':
        interpolate = OrdinaryKriging(x, y, z,
                             variogram_model=variogram,
                             pseudo_inv_type="pinvh")
        
    return interpolate

def krig2D_param(x,y,z):
    param_dict2d = {
                "method": ["ordinary","universal"],
                "variogram_model": ["linear", "power", "gaussian", "spherical"],
                "pseudo_inv_type":["pinvh"]}

    train = np.vstack((np.asarray(x), np.asarray(y)))
    train = train.T
    

    estimator = GridSearchCV(Krige(), param_dict2d, verbose=True, return_train_score=True, scoring = "neg_root_mean_squared_error")
    estimator.fit(X=train, y=z)

    if hasattr(estimator, "best_score_"):
        print("RMSE = {:.3f}".format(estimator.best_score_))
        print("best_params = ", estimator.best_params_)
        print("\nCV results::")
    if hasattr(estimator, "cv_results_"):
        for key in ["mean_test_score",
                    "mean_train_score",
                    "param_method",
                    "param_variogram_model"
                    ]:
            print(" - {} : {}".format(key, estimator.cv_results_[key]))
    
    return estimator.best_params_


list_csv = []
for file in glob.glob(os.path.join(root, 'PM\daily_jointPM') + '\*.csv'):
    list_csv.append(file)

list_npz = []
for file in glob.glob(os.path.join(root, 'PM\daily_PM') + '\*.npz'):
    list_npz.append(file)
    

mask_path = os.path.join(root, "Tehran_mask.npz")
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]

top_left_lat = 35.8318 
top_left_lon = 51.0725
bottom_right_lat = 35.5873
bottom_right_lon = 51.6233  
# 51.0725, 51.6233, 35.5873, 35.8318   
grid_lat = np.linspace( top_left_lat, bottom_right_lat, mask.shape[0])
grid_lon = np.linspace(top_left_lon, bottom_right_lon, mask.shape[1])
    
long_data , lat_data = np.meshgrid(grid_lon, grid_lat)

for i in range(len(list_npz)):
    print(list_npz[i])
    Date = list_npz[i][-11:-4]
    loaded_pm = np.load(list_npz[i])
    pm = loaded_pm["pm"]
    pm[pm==-9999]
    
    df = pd.read_csv(list_csv[i])
    x = df["long"].tolist()
    y = df["lat"].tolist()
    z = df["PM"].tolist()
    # param = krig2D_param(x,y,z)
    
    interpolate = krig2D(x,y,z, 'universal', 'gaussian')
    idx= np.argwhere(pm==-9999)


    for i in range(len(idx)):
        pm[idx[i,0],idx[i,1]], _ = interpolate.execute("points", long_data[idx[i,0],idx[i,1]], lat_data[idx[i,0],idx[i,1]])

    # export_path = os.path.join(root, 'PM\daily_krgPM', "daily_int_"+Date+".npz")
    # np.savez_compressed(export_path, pm=pm)
            
        
        
        


        
        

        
    
        
    
                
            
            

            
        
    
        

        
        
        
        
        
    

    



