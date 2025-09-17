# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:13:52 2020
Interpolate (climate) data at the time of MODIS pass
@author: Hossein Bagheri

"""
import pandas as pd
import os
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import numpy as np
from datetime import timedelta, date
from sklearn import preprocessing
# parameters
ini = 2013
end = 2019
band = "55"
grid_size = 3
met_name = "met"     

#LIST = ['d2m', 't2m', 'blh', 'lai_hv', 'lai_lv', 'sp', 'tp', 'ws10', 'wd10', cdir, uvb]
LIST = ['d2m']
MV = ["spherical", "universal"] # semi-variogram and method of kriging 
ref = 2000

root_path = "G:\project\AQ\Data"
fileName = "AOD"+"_g"+str(grid_size)+"_"+band+".csv"
path = os.path.join(root_path,"AOD_feature",fileName)
    

dfaod = pd.read_csv(path)
dfaod = dfaod.dropna()
dfaod = dfaod.reset_index(drop=True)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
    
df2 = dfaod.sort_values(["date", "station"], axis=0, ascending=[True, True])
# df2 = df2.reset_index()
Interp = []
idx = []
for year in range(2013, end+1):

    pathMETh  = os.path.join(root_path, "MET", "daily_"+str(year-ref)+met_name+".csv")
    df1 = pd.read_csv(pathMETd)
    df1 = df1.sort_values(["Sdate"], axis=0, ascending=[True])


    print (LIST)
    for var in LIST:
        print(var)
        start_date = date(year, 1, 1)
        end_date = date(year+1, 1, 1)
        for single_date in daterange(start_date, end_date):
            if str(single_date) not in df2.date.unique():
                continue
            print(single_date)
            sub_df1 = df1.loc[df1['Sdate'] == str(single_date)]
            sub_df2 = df2.loc[df2['date'] == str(single_date)] 
            val = sub_df1[var].values
            
    
            Xtrain=sub_df1[["latitude", "longitude"]].values
            # train data normalization
            scaler = preprocessing.MinMaxScaler()
            # #scaler = preprocessing.StandardScaler()
            Xtrain_scaled = scaler.fit_transform(Xtrain)
            xtrain = Xtrain_scaled[:, 0]
            ytrain = Xtrain_scaled[:, 1]

            target_scaler = preprocessing.MinMaxScaler()
            # #target_scaler = preprocessing.StandardScaler()
            val= target_scaler.fit_transform(val.reshape(-1, 1))
      
    
            Xtest=sub_df2[["lat", "long"]].values
            idx.append(sub_df2.index)

            Xtest_scaled = scaler.transform(Xtest)
            xtest = Xtest_scaled[:, 0]
            ytest = Xtest_scaled[:, 1]
            
            method = MV[1]
            variogram = MV[0]
    
            if method == 'universal':
                UK = UniversalKriging(xtrain, ytrain, val,
                                          variogram_model=variogram,
                                          pseudo_inv_type="pinvh")
                k, ss = UK.execute("points", xtest, ytest)
                Interp.append(target_scaler.inverse_transform(k.reshape(-1, 1)))
                
                
                #Interp.append(k3d2.reshape(-1, 1))
            else:
                OK = OrdinaryKriging(xtrain, ytrain, val,
                                          variogram_model=variogram,
                                          pseudo_inv_type="pinvh")
                k, ss = OK.execute("points", xtest, ytest)
                Interp.append(target_scaler.inverse_transform(k.reshape(-1, 1)))
                
idxf = []
for x in idx:
    for y in x:
        idxf.append(y)
                  

print(sum(idxf == df2.index)==len(df2.index))
df2[var]=pd.Series(np.concatenate(Interp, axis=0).reshape(-1)) 
  
df2.to_csv(os.path.join(path), index=False)





