# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:02:20 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
AOD data split into train, validation, test
Python 3, (env: base)
"""


import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_validate


root_path = "C:\project\AQ0\Data\AOD_feature"
name = 'data' + ".csv"
nametrain_out = 'train' + ".csv"
nametest_out = 'test' + ".csv"

path = os.path.join(root_path, name)
df = pd.read_csv(path)


X = df[['station', 'lat', 'long', 'AODm', 'Prob_bestm',
        'Prob_medm', 'month', 'd2m', 't2m','blh',
        'sp', 'lai_hv', 'lai_lv', 'wd10', 'cdir', 'uvb',
        'u', 'nAODm', 'DOY']]
y = df[['PMc']]
scaler = MinMaxScaler()



# cv = KFold(n_splits=5, random_state=1, shuffle=True)

# # from xgboost import XGBRFRegressor
# from sklearn.linear_model import LinearRegression


# MSE = []
# MAE = []
# R = []
# i = 0
# for train_index, test_index in cv.split(X):
#   print("loop:", i)
#   # print( "TRAIN:", train_index, "TEST:", test_index)
#   dftrain = df.loc[train_index]
#   dftest = df.loc[test_index]
#   Xtrain = dftrain[['nAODm', 'lat', 'long','d2m', 't2m', 'blh',
#                   'sp', 'lai_hv', 'ws10', 'wd10', 'uvb','RH',
#                   'DOY']]
#   ytrain = dftrain[['PMc']]
#   Xtest = dftest[['nAODm', 'lat', 'long','d2m', 't2m', 'blh',
#                 'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 'RH',
#                 'DOY']]
#   ytest = dftest[['PMc']]
#   Xtrain = scaler.fit_transform(Xtrain)
#   Xtest = scaler.fit_transform(Xtest)
#   model = LinearRegression()
#   model.fit(Xtrain, ytrain)
#   dftest["y_pred"] = model.predict(Xtest)

#   dftest["PMp"] = dftest.apply(NominalPM, axis = 1)
#   rmse = np.round(mean_squared_error(dftest["PMp"], dftest["PM2.5"], squared=False),2)
#   mae = np.round(mean_absolute_error(dftest["PM2.5"], dftest["PMp"]),2)
#   print ("RMSE: ", rmse)
#   print ("MAE: ", mae)
#   from scipy.stats import pearsonr
#   corr, _ = pearsonr(dftest["y_pred"], dftest["PMc"])
#   print('Pearsons R2 correlation: %.2f' % corr**2)
#   MSE.append(rmse)
#   MAE.append(mae)
#   R.append(corr**2)
#   i=i+1
  
  
  
  
  




# i = 0
# for train_index, test_index in cv.split(X):
#   print("loop:", i)

#   if i == : #3
#     print( "TRAIN:", train_index, "TEST:", test_index)
    
#     break
#   else:
#     i+=1
# train = df.loc[train_index]
# out_path = os.path.join(root_path, nametrain_out)
# train.to_csv(out_path , index=False)

# test = df.loc[test_index]
# out_path = os.path.join(root_path, nametest_out)
# test.to_csv(out_path , index=False)


# plt.figure()
# plt.hist(train['PMc'])
# plt.title('train')
# plt.show()

# plt.figure()
# plt.hist(test['PMc'])
# plt.title('test')
# plt.show()
# out_path = os.path.join(root_path, 'train.csv')
# train.to_csv(out_path , index=False)

# out_path = os.path.join(root_path, 'test.csv')
# test.to_csv(out_path , index=False)
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 123)
# Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size = 0.2, random_state = 123)

dftrain = df.loc[Xtrain.index]
# dfval = df.loc[Xval.index]
dftest = df.loc[Xtest.index]

Xtrain = scaler.fit_transform(Xtrain)
# Xval = scaler.fit_transform(Xval)
Xtest = scaler.fit_transform(Xtest)

from sklearn.linear_model import LinearRegression
# from lightgbm import LGBMRegressor

# model = LGBMRegressor()
model = LinearRegression()


model.fit(Xtrain, ytrain)


# # Validation
# dfval["y_pred"] = model.predict(Xval)

# # convert predictions to nominal values
# def NominalPM(row):
#     return row["y_pred"]/((1-row["RH"])**(-1))
# dfval["PMp"] = dfval.apply(NominalPM, axis = 1)

# print ("RMSE: ", np.round(mean_squared_error(dfval["PMp"], dfval["PM2.5"], squared=False),2))
# print ("MAE: ", np.round(mean_absolute_error(dfval["PM2.5"], dfval["PMp"]),2))
# from scipy.stats import pearsonr
# corr, _ = pearsonr(dfval["y_pred"], dfval["PMc"])
# print('Pearsons R2 correlation: %.2f' % corr**2)

########## Train
print('------------------Train------------------')
dftrain["y_pred"] = model.predict(Xtrain)

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

########## Test
print('------------------Test------------------')
dftest["y_pred"] = model.predict(Xtest)
dftest["PMp"] = dftest.apply(NominalPM, axis = 1)
RMSE = np.round(mean_squared_error(dftest["PM2.5"], dftest["PMp"], squared=False),2)
MAE = np.round(mean_absolute_error(dftest["PM2.5"], dftest["PMp"]),2)
print ("RMSE: ", RMSE)
print ("MAE: ", MAE)
corr, _ = pearsonr(dftest["y_pred"], dftest["PMc"])
print('Pearsons R2 correlation: %.2f' % corr**2)



out_path = os.path.join(root_path, 'train.csv')
dftrain.to_csv(out_path , index=False)

out_path = os.path.join(root_path, 'test.csv')
dftest.to_csv(out_path , index=False)

plt.figure()
plt.hist(dftrain['PMc'])
plt.title('train')
plt.show()

plt.figure()
plt.hist(dftest['PMc'])
plt.title('test')
plt.show()



