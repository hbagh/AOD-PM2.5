# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 09:57:25 2020
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Join all PM-AOD and meteorological data-frames collected in different years
Phyton3 (env: base)
"""
# parameters
grid_size = 3 

band = "55"
ini = 2013
end = 2019
ref_year = 2000

import pandas as pd
import os
from datetime import timedelta, date
import numpy as np
import datetime


root_path = "G:\project\AQ\Data"


name_out = "AOD_PM" + "_g"+str(grid_size) + "_" + band + ".csv"

namePM = "PM.csv"
pathPM = os.path.join(root_path,"Ground_measurements", namePM)
nameAOD = "AOD"+"_g"+str(grid_size)+"_"+band+".csv" 
pathAOD = os.path.join(root_path,"AOD_feature",nameAOD)

dfPM = pd.read_csv(pathPM)
dfAOD = pd.read_csv(pathAOD)

aodPM = []



def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

for i in range(23):
    dfPMs = dfPM.loc[dfPM['station'] == i]
    dfAODs = dfAOD.loc[dfAOD['station'] == i]     
    for year in range(ini, end+1):
        start_date = date(year, 1, 1)
        end_date = date(year+1, 1, 1)
           
        for single_date in daterange(start_date, end_date):
            print("date: ", single_date, 'station:',  i)
            dfPMd = dfPMs.loc[dfPMs['date'] == str(single_date)]
            dfAODd = dfAODs.loc[dfAODs['date'] == str(single_date)]
            if str(single_date) not in dfAODd.date.unique():
                continue
            elif str(single_date) not in dfPMd.date.unique():
                aodPM.append(np.c_[dfAODd.to_numpy() , np.nan])
            else:
                pm = dfPMd.to_numpy()[:,1]
                aodPM.append(np.c_[dfAODd.to_numpy() , pm])

data = np.vstack(aodPM)

df = pd.DataFrame(data, columns = ['station', 'lat', 'long', 'AODm', 'Prob_bestm',
                                   'Prob_medm', 'std', 'No of pixels', 'grid_size',
                                   'band', 'Jdate', 'Sdate', 'date', 'month','d2m',
                                   't2m', 'blh', 'sp', 'lai_hv', 'lai_lv', 'ws10', 'wd10', 
                                   'cdir', 'uvb', "PM2.5"])
        
def RH(row):
    return ((row["d2m"]-0.1*row["t2m"]+112)/(112+0.9*row["t2m"]))**(1/0.125)

def kel2cel(k):
    return k-273.5

df["t2mC"] = [kel2cel(t) for t in df["t2m"]]
df["d2mC"] = [kel2cel(td) for td in df["d2m"]]
df['RH'] = df.apply(RH, axis = 1)
df = df.drop(columns = ["t2mC", "d2mC"])


def CorrectPM(row):
    return row["PM2.5"]*((1-row["RH"])**(-1))
df["PMc"] = df.apply(CorrectPM, axis = 1)

def NormalAOD(row):
    return row["AODm"]/(row["blh"]/1000)
df["nAODm"] = df.apply(NormalAOD, axis = 1)

def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)
df["DOY"] = [datestdtojd(d) for d in df["date"]]

df = df[df["No of pixels"] > 2]
df = df[df["std"] < 500]
Count_df= df.shape[0]

df = df.drop(columns=["Jdate", "Sdate", "std", "No of pixels", "band", "grid_size"])
df.dropna(inplace=True)


mask = np.logical_or(df["PM2.5"] > 0, df["PM2.5"] == 0)
df = df[mask]

# First quartile (Q1) 
Q1 = np.percentile(df["PM2.5"], 25, interpolation = 'midpoint')  
# Third quartile (Q3) 
Q3 = np.percentile(df["PM2.5"], 75, interpolation = 'midpoint')  
IQR = Q3 - Q1
lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR
print("lower band:", lb, ("upper band: ", ub))
# # sig3 = np.mean(data["PM2.5"]) + np.std(data["PM2.5"])*1
maskQ = np.logical_and(df["PM2.5"]>lb, df["PM2.5"]<ub) 
df= df[maskQ]

print('Percentage of available data:', round((df.shape[0]/(dfPM.dropna().shape[0]))*100), "%")
out_path = os.path.join(root_path, "final_data", name_out)
df.to_csv(out_path , index=False)

