# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:16:56 2021

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Linar regression (MV) for regression of AOD-PM using in-situ RH
Python 3, (env: base)
"""
# parameters
varAOD = 'nAODm'
varPM = 'PMc'

import numpy as np
import pandas as pd 
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
 

features = [varAOD, 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
                  'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'u', 'DOY']

root_path = "C:\Project\AQ0\Data\AOD_feature"
name = 'data' + ".csv"



import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


root_path = "C:\project\AQ0\Data\AOD_feature"
name = 'data' + ".csv"

path = os.path.join(root_path, name)
dftrain = pd.read_csv(path)


X = dftrain[[varAOD, 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
                  'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'u', 'DOY']]
y = dftrain[['PMc']]
scaler = MinMaxScaler()


X = scaler.fit_transform(X)


from sklearn.linear_model import LinearRegression

model = LinearRegression()


model.fit(X, y)


########## Train
print('------------------Train------------------')
dftrain["y_pred"] = model.predict(X)

def NominalPM(row):
    return row["y_pred"]/((1-row["u"])**(-1))
dftrain["PMp"] = dftrain.apply(NominalPM, axis = 1)
RMSE = np.round(mean_squared_error(dftrain["PM2.5"], dftrain["PMp"], squared=False),2)
MAE = np.round(mean_absolute_error(dftrain["PM2.5"], dftrain["PMp"]),2)
print ("RMSE: ", RMSE)
print ("MAE: ", MAE)
from scipy.stats import pearsonr
corr, _ = pearsonr(dftrain["y_pred"], dftrain["PMc"])
print('Pearsons R2 correlation: %.2f' % corr**2)




