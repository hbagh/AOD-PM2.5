# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 03:27:28 2020
Last update on 30/10/2020

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Seasonal regression of AOD collected by Terra and Aqua (extracted from MAIAC products)

"""
# parameters

band = "55" # or 55
ini = 2013
end = 2014




import os
import glob
import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd



if band  == "55":
    DATAFIELD_NAME = 'Optical_Depth_055'
else:
    DATAFIELD_NAME = 'Optical_Depth_047'
print ("band for AOD retrieval: ", DATAFIELD_NAME)


def j2month(jdate):
    # converting to date
    res = int(datetime.strptime(jdate, "%Y%j").strftime("%m"))
    return res

# Opens the data HDF file and returns as a dataframe
def read_dataset(DATAFIELD_NAME, FILEPATH):
    # Read dataset.
    hdf = SD(FILEPATH, SDC.READ)
    hdf_attr = hdf.attributes(full=1)
    # number of orbits
    orb_amount = hdf_attr["Orbit_amount"]
    orb = orb_amount[0]
    orb_time_stamp = hdf_attr["Orbit_time_stamp"] 
    orb_time = orb_time_stamp[0].split()
    data3D = hdf.select(DATAFIELD_NAME)
    Q3D = hdf.select("AOD_QA")
    # Read attributes.
    attrs = data3D.attributes(full=1)
    fva=attrs["_FillValue"]
    _FillValue = fva[0]
    sfa=attrs["scale_factor"]
    scale_factor = sfa[0]
    return data3D, Q3D, _FillValue, scale_factor, orb, orb_time

vTtotal = []
vAtotal = []
dicAT = []
dicTA = []

vTcold = []
vAcold = []
vTwarm = []
vAwarm = []

for year in range (ini, end+1):
    print ("AOD for " + str(year) + "\n")
    Name = "AOD" + str(year-2000)
    directory = os.path.join('C:\Project\AQ\Data\AOD' , Name, "AOD" + str(year))

    for file in glob.glob(directory + '\*.hdf'):
        FILEPATH = file
        print(FILEPATH)
        jdate = FILEPATH[-36:-29]
        month = j2month(jdate)
        
        data3D, Q3D, _FillValue, scale_factor, orb, orb_time = read_dataset(DATAFIELD_NAME, FILEPATH)
        dataT = []
        dataA = []
 

        for k in range (orb):
             plat = orb_time[k][-1]
             data = data3D[k,:,:].astype(float)
             data[data==_FillValue] = np.nan
             
             if plat == "T":
                 dataT.append(data)
             else: 
                 dataA.append(data)

                 
        meanT = np.nanmean(dataT, axis=0)
        meanA = np.nanmean(dataA, axis=0)
        vT0 = np.nanmean(meanT)
        vA0 = np.nanmean(meanA)
        if (np.isnan(vT0) == False) and (np.isnan(vA0) == False) and (month > 2 and month< 9):
            vTwarm.append(vT0)
            vAwarm.append(vA0)

        if (np.isnan(vT0) == False) and (np.isnan(vA0) == False) and (month > 8 and month < 13 or month == 1 or month == 2):
            vTcold.append(vT0)
            vAcold.append(vA0)
        
    #     #"Aqua2Terra: Aqua:X, Terra:y, Linear regression"
    # print("ATwarm: Aqua2Terra: Aqua:X, Terra:y, Linear regression")
    # X = np.asarray(vAwarm).reshape(-1,1)
    # y = np.asarray(vTwarm).reshape(-1,1)
    # modelAT = LinearRegression().fit(X,y)
    # rATwarm = modelAT.score(X, y)
    # print('correlation (ATwarm):', rATwarm)
    # dicAT = dict({"coefAT": modelAT.coef_[0,0], "intcAT": modelAT.intercept_[0], "rAT": rATwarm})
        
        
    #     # Terra2Aqua: Aqua:y, Terra:X, Linear regression
    # print("TA: Terra2Aqua: Terra:X, Aqua:y, Linear regression")
    # X = np.asarray(vTwarm ).reshape(-1,1)
    # y = np.asarray(vAwarm).reshape(-1,1)
    # modelTA = LinearRegression().fit(X,y)
    # rTAwarm = modelTA.score(X, y)
    # print('correlation (TA):', rTAwarm)
    # dicTA = dict({"coefTA": modelTA.coef_[0,0], "intcTA": modelTA.intercept_[0], "rTA": rTAwarm})
    

    #     #"Aqua2Terra: Aqua:X, Terra:y, Linear regression"
    # print("ATcold: Aqua2Terra: Aqua:X, Terra:y, Linear regression")
    # X = np.asarray(vAcold).reshape(-1,1)
    # y = np.asarray(vTcold).reshape(-1,1)
    # modelAT = LinearRegression().fit(X,y)
    # rATcold = modelAT.score(X, y)
    # print('correlation (ATwarm):', rATcold)
    # dicAT = dict({"coefAT": modelAT.coef_[0,0], "intcAT": modelAT.intercept_[0], "rAT": rATcold})
        
        
    #     # Terra2Aqua: Aqua:y, Terra:X, Linear regression
    # print("TA: Terra2Aqua: Terra:X, Aqua:y, Linear regression")
    # X = np.asarray(vTcold ).reshape(-1,1)
    # y = np.asarray(vAcold).reshape(-1,1)
    # modelTA = LinearRegression().fit(X,y)
    # rTAcold = modelTA.score(X, y)
    # print('correlation (TA):', rTAcold)
    # dicTA = dict({"coefTA": modelTA.coef_[0,0], "intcTA": modelTA.intercept_[0], "rTA": rTAcold})
    # # nameAT = directory + "_AT_reg_" + "AOD_" + band + ".json"
    # # with open(nameAT, 'w') as fout:
    # #     json.dump(dicAT, fout)
    # # nameTA = directory + "_TA_reg_" + "AOD_" + band + ".json"   
    # # with open(nameTA, 'w') as fout:
    # #     json.dump(dicTA, fout)
        
    # # vTtotal.append(vT)
    # # vAtotal.append(vA)

    


Xc = np.hstack(vAcold).reshape(-1,1)
yc = np.hstack(vTcold).reshape(-1,1)  
       
# plt.figure()
# plt.scatter(X,y)
# plt.show()


df = pd.DataFrame(np.hstack((Xc,yc)),columns=['X','y'])



# regression plot
fig, ax = plt.subplots(1, figsize=(4.5, 4))
# plt.rc('font', family='serif')
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Arial'}
ax = sb.regplot(df["X"], df["y"], data =df,
            line_kws={"color": "red"}, scatter=True, fit_reg=True, color = 'blue', marker='+')
plt.legend(['R2 = {:.2f}'.format(np.round(0.72, 2))])


plt.xlabel('AOD(Terra)', **hfont)
plt.ylabel('AOD(Aqua)', **hfont)
plt.xticks(**hfont)
plt.yticks(**hfont)


Xw = np.hstack(vAwarm).reshape(-1,1)
yw = np.hstack(vTwarm).reshape(-1,1)  
       
# plt.figure()
# plt.scatter(X,y)
# plt.show()


df = pd.DataFrame(np.hstack((Xw,yw)),columns=['X','y'])



# regression plot
fig, ax = plt.subplots(1, figsize=(4.5, 4))
# plt.rc('font', family='serif')
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Arial'}
ax = sb.regplot(df["X"], df["y"], data =df,
            line_kws={"color": "red"}, scatter=True, fit_reg=True, color = 'blue', marker='+')
plt.legend(['R2 = {:.2f}'.format(np.round(0.72, 2))])


plt.xlabel('AOD(Terra)', **hfont)
plt.ylabel('AOD(Aqua)', **hfont)
plt.xticks(**hfont)
plt.yticks(**hfont)
# #"Aqua2Terra: Aqua:X, Terra:y, Linear regression"
# print("Yearly, AT: Aqua2Terra: Aqua:X, Terra:y, Linear regression")
# modelAT = LinearRegression().fit(X,y)
# rAT = modelAT.score(X, y)
# print('Yearly, correlation (AT total):', rAT)
# dicATtotal = dict({"coefAT": modelAT.coef_[0,0], "intcAT": modelAT.intercept_[0], "rAT": rAT} )
        
        
# # Terra2Aqua: Aqua:y, Terra:X, Linear regression
# print("Yeraly, TA: Terra2Aqua: Terra:X, Aqua:y, Linear regression")
# X = np.hstack(vTtotal).reshape(-1,1)
# y = np.hstack(vAtotal).reshape(-1,1)  

# modelTA = LinearRegression().fit(X,y)
# rTA = modelTA.score(X, y)
# print('Yearly, correlation (TA total):', rTA)
# dicTAtotal  = dict({"coefTA": modelTA.coef_[0,0], "intcTA": modelTA.intercept_[0], "rTA": rTA})

# direc = 'C:\Project\AQ\Data\AOD'

# nameATtotal = os.path.join(direc, "AT_total_reg_" + "AOD_" + band + ".json")   
# with open(nameATtotal, 'w') as fout:
#     json.dump(dicATtotal, fout)
    
# nameTAtotal = os.path.join(direc, "TA_totalreg_" + "AOD_" + band + ".json")  
# with open(nameTAtotal, 'w') as fout:
#     json.dump(dicTAtotal, fout)

print('\nAll valid files have been processed')    

 



