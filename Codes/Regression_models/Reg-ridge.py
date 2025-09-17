# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 15:28:49 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Ridge regression AOD-PM
Python 3, (env: base)
"""

# parameters
varAOD = 'nAODm'
varPM = 'PMc'


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings # supress warnings

warnings.filterwarnings('ignore')

features = [varAOD, 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
                  'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'RH', 'DOY']

root_path = 'G:\project\AQ0\Data'
name = 'train' + '.csv'
path = os.path.join(root_path, name)

dftrain = pd.read_csv(path)

Xtrain = dftrain[features]


ytrain = dftrain.loc[:,[varPM]]
ytrain = ytrain.to_numpy()
ytrain = np.ravel(ytrain)
scaler = MinMaxScaler()
Xstrain = scaler.fit_transform(Xtrain)

# Model
from sklearn.linear_model import Ridge
model = Ridge(alpha = 0.1)


name = 'test' + ".csv"
path = os.path.join(root_path, name)

dftest = pd.read_csv(path)
Xtest = dftest[features]
ytest = dftest.loc[:,[varPM]]
ytest = ytest.to_numpy()
ytest = np.ravel(ytest)
Xstest = scaler.fit_transform(Xtest)


model.fit(Xstrain, ytrain)

########## Train
print('------------------Train------------------')
dftrain["y_pred"] = model.predict(Xstrain)

def NominalPM(row):
    return row["y_pred"]/((1-row["RH"])**(-1))
dftrain["PMp"] = dftrain.apply(NominalPM, axis = 1)
RMSE = np.round(mean_squared_error(dftrain["PM2.5"], dftrain["PMp"], squared=False),2)
MAE = np.round(mean_absolute_error(dftrain["PM2.5"], dftrain["PMp"]),2)
print ("RMSE: ", RMSE)
print ("MAE: ", MAE)
from scipy.stats import pearsonr
corr, _ = pearsonr(dftrain["y_pred"], dftrain["PMc"])
print('Pearsons R2 correlation: %.2f' % corr**2)

########## Test
print('------------------Test------------------')
dftest["y_pred"] = model.predict(Xstest)
dftest["PMp"] = dftest.apply(NominalPM, axis = 1)
RMSE = np.round(mean_squared_error(dftest["PM2.5"], dftest["PMp"], squared=False),2)
MAE = np.round(mean_absolute_error(dftest["PM2.5"], dftest["PMp"]),2)
print ("RMSE: ", RMSE)
print ("MAE: ", MAE)
corr, _ = pearsonr(dftest["y_pred"], dftest["PMc"])
print('Pearsons R2 correlation: %.2f' % corr**2)

'''
------------------Train------------------
RMSE:  10.96
MAE:  8.64
Pearsons R2 correlation: 0.61
------------------Test------------------
RMSE:  12.42
MAE:  9.84
Pearsons R2 correlation: 0.59
'''