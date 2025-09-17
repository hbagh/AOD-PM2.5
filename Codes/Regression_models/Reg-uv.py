# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:38:13 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
univariate regression AOD-PM
Python 3, (env: base)
"""


varAOD = 'nAODm'
varPM = 'PMc'


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings # supress warnings
warnings.filterwarnings('ignore')


root_path = 'G:\project\AQ0\Data'
name = 'train' + '.csv'
path = os.path.join(root_path, name)

dftrain = pd.read_csv(path)
Xtrain = dftrain[[varAOD]]
ytrain = dftrain.loc[:,['PMc']]
ytrain = ytrain.to_numpy()
ytrain = np.ravel(ytrain)
scaler = MinMaxScaler()
Xstrain = scaler.fit_transform(Xtrain)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

name = 'test' + ".csv"
path = os.path.join(root_path, name)

dftest = pd.read_csv(path)
Xtest = dftest[[varAOD]]
ytest = dftest.loc[:,['PMc']]
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
RMSE:  13.85
MAE:  10.96
Pearsons R2 correlation: 0.40
------------------Test------------------
RMSE:  15.78
MAE:  12.34
Pearsons R2 correlation: 0.41
'''