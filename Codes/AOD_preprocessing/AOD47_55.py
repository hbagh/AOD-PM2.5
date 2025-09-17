# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:39:57 2020
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Compare AOD47 and 55
Phyton3 (env: base)
"""

# parameters
grid_size = 3 # size of averaging window for calculating AOD or other products
band = "47"
varAOD = 'nAODm'
varPM = 'PMc'

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



root_path = "G:\project\AQ\data-Iran_Tehran"
name47 = "fAOD_PM" + "_g"+str(grid_size) + "_" + "47" + ".csv"
path47 = os.path.join(root_path, "final_data", name47)
df47 = pd.read_csv(path47)


X47 = df47.loc[:,[varAOD, 'lat', 'long', 'month', 'd2m', 't2m','blh',
        'sp', 'lai_hv', 'lai_lv', 'ws10', 'wd10', 'cdir', 'uvb', 
        'RH', 'DOY']]
y47 = df47.loc[:,[varPM]]

scaler = MinMaxScaler()
Xs47 = scaler.fit_transform(X47)
#ys = scaler.fit_transform(y)
name55 = "fAOD_PM" + "_g"+str(grid_size) + "_" + "55" + ".csv"
path55 = os.path.join(root_path, "final_data", name55)
df55 = pd.read_csv(path55)


X55 = df55.loc[:,[varAOD, 'lat', 'long', 'month', 'd2m', 't2m','blh',
        'sp', 'lai_hv', 'lai_lv', 'ws10', 'wd10', 'cdir', 'uvb', 
        'RH', 'DOY']]
y55 = df55.loc[:,[varPM]]

scaler = MinMaxScaler()
Xs55 = scaler.fit_transform(X55)
dif=df55[varAOD]-df47[varAOD]
import matplotlib.pyplot as plt
plt.hist(dif, bins = 100)
plt.show()