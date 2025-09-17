# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:38:59 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Extra Trees for regression of AOD-PM
Python 3, (env: base)
"""
# parameters
varAOD = 'nAODm'
varPM = 'PMc'

import numpy as np
import pandas as pd 
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.model_selection import GridSearchCV

features = [varAOD, 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
                  'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'RH', 'DOY']

root_path = "G:\project\AQ0\Data"
name = 'train' + ".csv"
path = os.path.join(root_path, name)
dftrain = pd.read_csv(path)
Xtrain = dftrain[features]
ytrain = dftrain.loc[:,['PMc']]
ytrain = ytrain.to_numpy()
ytrain = np.ravel(ytrain)
scaler = MinMaxScaler()
Xstrain = scaler.fit_transform(Xtrain)

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()

######## Grid search-Kfold#####

# cv = KFold(n_splits=10, random_state=1, shuffle=True)
# distributions = dict(n_estimators = [100, 500, 1000], 
#                      max_depth=[3, 5, 7, 8, 10],
#                      max_features=[0.5, 0.7, 0.8, 1])

# reg = GridSearchCV(model, distributions, verbose=2)
# search = reg.fit(Xs, y)
# print('----------------------------')
# print(search.best_params_)
# print('----------------------------')
# # {'max_depth': 10, 'max_features': 0.8, 'n_estimators': 1000}

name = 'test' + ".csv"
path = os.path.join(root_path, name)
dftest = pd.read_csv(path)
Xtest = dftest[features]
ytest = dftest.loc[:,['PMc']]
ytest = ytest.to_numpy()
ytest = np.ravel(ytest)
Xstest = scaler.fit_transform(Xtest)


model = ExtraTreesRegressor(n_estimators= 1000,#search.best_params_['n_estimators'],
                     max_depth= 10,#search.best_params_['max_depth'],
                     max_features= 0.8,#search.best_params_['max_features']
                             )

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
RMSE:  8.63
MAE:  6.82
Pearsons R2 correlation: 0.80
------------------Test------------------
RMSE:  9.77
MAE:  7.8
Pearsons R2 correlation: 0.68
'''
