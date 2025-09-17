# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:07:20 2020
Last update on 30/10/2020

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
AOD extraction from MODIs products
Python2.7 (env: aq-py27)

"""
# parameters
grid_size = 7 # size of averaging window for calculating AOD or other products
band = "55" # or 55  
ini = 2013
end = 2019
scale_factor = 0.001
_FillValue = -28672




import os
import glob
import numpy as np
import pandas as pd
import datetime




# position of stations
path = os.path.join("C:\Project\AQ0",
                    "Data",
                    "station_location.txt")
station = np.loadtxt(path)

if band  == "55":
    DATAFIELD_NAME = 'Optical_Depth_055'
else:
    DATAFIELD_NAME = 'Optical_Depth_047'
print ("band for AOD retrieval: ", DATAFIELD_NAME)


# define a window for AOD extraction
th1=int((grid_size-1)*0.5)
th2 = int(th1+1)


def product(x, y, data, Qbest, Qmed, scale_factor, _FillValue, grid_size):
    #calculates mean, median, stdev in a xbyx grid around nearest point to entered location
    if x < th1:
        x+=1
    if x > data.shape[0]-th2:
        x-=2
    if y < th1:
        y+=1
    if y > data.shape[1]-th2:
        y-=2
    x_by_x=data[x-th1:x+th2, y-th1:y+th2]
    qbest = Qbest[x-th1:x+th2, y-th1:y+th2]
    qmed = Qmed[x-th1:x+th2, y-th1:y+th2]
    x_by_x=x_by_x.astype(float)
    mask = x_by_x==float(_FillValue)
    x_by_x[mask]=np.nan
    qbest[mask]=np.nan 
    qmed[mask]=np.nan 
    Num = np.count_nonzero(~np.isnan(x_by_x))
    qbestm = np.nanmean(qbest)
    qmedm = np.nanmean(qmed)
    aodm = np.nanmean(x_by_x)
    aodm=round(aodm*scale_factor, 5)
    std=np.nanstd(x_by_x)
    return aodm, qbestm, qmedm, std, Num
AOD_station = []  
for year in range (ini, end+1):
    print ("AOD for " + str(year) + "\n")
    Name = "AOD" + str(year-2000)
    root_dir = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD' , Name)
    path_aod = os.path.join(root_dir, "aod"+band)
    path_Qbest = os.path.join(root_dir, "Qbest"+band)
    path_Qmed = os.path.join(root_dir, "Qmed"+band)
    
    # AOD_station_new = []
    FILEPATH_aod = []
    for file in glob.glob(path_aod + '\*.npz'):
        FILEPATH_aod.append(file)
        # print(file)
        
    FILEPATH_Qb = []
    for file in glob.glob(path_Qbest + '\*.npz'):
        FILEPATH_Qb.append(file)
        # print(file)
        # loaded_Qb = np.load(file)
        # Qbest = loaded_Qb["qp"]
    FILEPATH_Qm = []
    for file in glob.glob(path_Qmed + '\*.npz'):
        FILEPATH_Qm.append(file)
    
    for i in range(len(FILEPATH_aod)):
        print(FILEPATH_aod[i])
        date = FILEPATH_aod[i][-14:-7]
        loaded_aod = np.load(FILEPATH_aod[i])
        aod = loaded_aod["a"]
        loaded_Qb = np.load(FILEPATH_Qb[i])
        Qbest = loaded_Qb["qp"]
        loaded_Qm = np.load(FILEPATH_Qm[i])
        Qmed = loaded_Qm["qp"]
        
        for i in range(station.shape[0]):
            x = int(station[i,3])
            y = int(station[i,4])
            aodm, qbestm, qmedm, std, Num = product(x, y, aod, Qbest, Qmed, scale_factor,
                                                    _FillValue, grid_size)
            AOD_station.append([i, station[i,1], station[i,2], aodm, qbestm, qmedm,
                                std, Num, grid_size, DATAFIELD_NAME, date])

AOD_array=np.asarray(AOD_station)
df = pd.DataFrame(AOD_array, columns = ["station", "lat", "long", "AODm", 
                                        "Prob_bestm", "Prob_medm", "std", "No of pixels",
                                        "grid_size", "band", "Jdate"])

def jdtodatestd (jdate):
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)
D=[]
for d in df["Jdate"]:
    d_new = jdtodatestd(d)
    # d_new = d_new.strftime('%Y-%m-%d')
    D.append(d_new)
df['Sdate'] = D
df['date'] = pd.to_datetime(df['Sdate'], errors='coerce')
df['month'] = df['date'].dt.month


output_name = "AOD"+"_g"+str(grid_size)+"_"+band+".csv"
output = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD', output_name)
# df.dropna(inplace=True)
df.to_csv(output, index=False)

print('\nAll valid files have been processed')    




