# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:36:47 2020
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
univariate regression AOD-PM
Python 3, (env: base)
"""


# df1 = df[:, df["Prob_bestm"].loc < 0.25]
# df2 = df[:, (df["Prob_bestm"].loc > 0.25 and df["Prob_bestm"].loc < 0.5)]

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
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
import warnings # supress warnings
warnings.filterwarnings('ignore')


root_path = "G:\project\AQ\data-Iran_Tehran"
name = "fAOD_PM" + "_g"+str(grid_size) + "_" + band + ".csv"
path = os.path.join(root_path, "final_data", name)
df = pd.read_csv(path)
df_size = df.shape[0] 

def uv_reg(X,y):
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    ys = scaler.fit_transform(y)
    cv_method = KFold(n_splits=10, random_state=1, shuffle=True)
    
    def PolynomialRegression(degree=1, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
    param_grid = {'polynomialfeatures__degree': [1]}
    scoring = ['r2','neg_mean_squared_error']
    gs_polyReg = GridSearchCV(PolynomialRegression(), 
                          param_grid, 
                          cv=cv_method,
                          verbose=1,  # verbose: the higher, the more messages
                          scoring=scoring,
                          refit='neg_mean_squared_error',
                          return_train_score=True)
    gs_polyReg.fit(Xs, y)
    df1["y_pred"] = cross_val_predict(gs_polyReg, Xs, y, cv=cv_method)
    
    def NominalPM(row):
        return row["y_pred"]/((1-row["RH"])**(-1))
    
    df1["PMp"] = df1.apply(NominalPM, axis = 1)
    
    print("optimal degree: ", gs_polyReg.best_params_)
    # print("Best score: ", gs_polyReg.best_score_)
    print ("RMSE: ", np.round(mean_squared_error(df1["PM2.5"], df1["PMp"], squared=False),2))
    print ("MAE: ", np.round(mean_absolute_error(df1["PM2.5"], df1["PMp"]),2))
    print ("R2: ", gs_polyReg.cv_results_['mean_test_r2'])
    return gs_polyReg.cv_results_['mean_test_r2'], np.round(mean_absolute_error(df1["PM2.5"], df1["PMp"]),2), np.round(mean_squared_error(df1["PM2.5"], df1["PMp"], squared=False),2) 

RMSEm = []
MAEm = []
Rm = []
Pm = []
for i in np.arange(0,1,0.25): 
    print("probability: {}".format(i))
    df1 = df[df["Prob_medm"] > i]
    
    X = df1[[varAOD]]
    y = df1[[varPM]]
    r2, mae, rmse= uv_reg(X, y)

    Pm.append(round((df1.shape[0]/df_size)*100))
    Rm.append(r2)
    MAEm.append(mae)
    RMSEm.append(rmse)
    
RMSEb = []
MAEb = []
Rb = []
Pb = []
# df1 = df[df["Prob_bestm"] <= 0.25]
for i in np.arange(0,1,0.25): 
    df1 = df[df["Prob_bestm"] > i]
    print("probability: {}".format(i))

    X = df1[[varAOD]]
    y = df1[[varPM]]
    r2, mae, rmse= uv_reg(X, y)

    Pb.append(round((df1.shape[0]/df_size)*100))
    Rb.append(r2)
    MAEb.append(mae)
    RMSEb.append(rmse)
    
import matplotlib.font_manager as font_manager

x = np.arange(0,1,0.25)
fig, ax = plt.subplots(1, figsize=(5, 5))
hfont = {'fontname':'Arial'}
font = font_manager.FontProperties(family='Arial')

qb=plt.scatter(x, Rb, s=Pb, c= 'red', alpha=1, marker = "o")

for i, txt in enumerate(Pb):
    ax.annotate(txt, (x[i]+0.005, Rb[i]+0.005))
qm=plt.scatter(x, Rm, s=Pm, c= 'blue', alpha=1, marker = "s")
for i, txt in enumerate(Pm):
    ax.annotate(txt, (x[i]+0.005, Rm[i]-0.0085))
plt.xticks([0, 0.25, 0.5, 0.75])

plt.legend((qb, qm), ("Best quality flag", "Medium qulaity flag"),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=10,
           prop=font)
plt.xlabel("Probability of quality", **hfont)
plt.ylabel("R2", **hfont)
plt.show()

