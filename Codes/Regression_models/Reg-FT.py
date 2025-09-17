# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:53:41 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Feature Transform for regression of AOD-PM
Python 3, (env: base)
"""

# parameters
grid_size = 3 # size of averaging window for calculating AOD or other products

band = "55"
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


root_path = "G:\project\AQ\data-Iran_Tehran"
name = "fAOD_PM" + "_g"+str(grid_size) + "_" + band + ".csv"
path = os.path.join(root_path, "final_data", name)
df = pd.read_csv(path)

X = df[[varAOD, 'lat', 'long', 'Prob_bestm','d2m', 't2m', 'blh',
       'sp', 'lai_hv', 'ws10', 'wd10', 'uvb', 
       'RH', 'DOY']]


print(X.columns)
y = df[[varPM]]
y = y.to_numpy()

scaler = MinMaxScaler()
Xs = scaler.fit_transform(X)
print(Xs.shape)
print(y.shape)


cv = KFold(n_splits=10, random_state=1, shuffle=True)
i = 0
for train_index, test_index in cv.split(X):
  print("loop:", i)
  if i == 6:
    print( "TRAIN:", train_index, "TEST:", test_index)
    break
  else:
    i+=1
Xtrain = Xs[train_index]
ytrain = y[train_index]

Xtest = Xs[test_index]
ytest = y[test_index]

print(Xtrain.shape)
print(ytrain.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
def PolynomialRegression(degree, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
model = PolynomialRegression(degree=2)
model.fit(Xtrain, ytrain)
# make predictions on the test set
dftest = df.loc[test_index]

import time
start = time.time()
# prediction on validation data
dftest["y_pred"] = model.predict(Xtest)
print("Time of execution:",  (time.time() - start))

def NominalPM(row):
    return row["y_pred"]/((1-row["RH"])**(-1))
dftest["PMp"] = dftest.apply(NominalPM, axis = 1)

print ("RMSE: ", np.round(mean_squared_error(dftest["PMp"], dftest["PM2.5"], squared=False),2))
print ("MAE: ", np.round(mean_absolute_error(dftest["PM2.5"], dftest["PMp"]),2))
from scipy.stats import pearsonr
corr, _ = pearsonr(dftest["y_pred"], dftest["PMc"])
print('Pearsons R2 correlation: %.3f' % corr**2)


import matplotlib.pyplot as plt
import seaborn as sb
# regression plot
fig, ax = plt.subplots()
# plt.rc('font', family='serif')
ax = sb.regplot(dftest["y_pred"], dftest["PMc"], data = df, 
            line_kws={"color": "red"}, scatter=False)
plt.legend(['R2 = {:.2f}'.format(np.round(corr**2, 2))])
plt.xlabel('Measured PM2.5')
plt.ylabel('Predicted PM2.5')


# plt.figure()
# plt.hist(y)
# plt.show()
from scipy.stats import gaussian_kde
x, y = dftest["y_pred"].to_numpy(), dftest["PMc"].to_numpy()


# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)


ax.scatter(x, y, c=z, s=7, edgecolor='', cmap='jet')
plt.show()





