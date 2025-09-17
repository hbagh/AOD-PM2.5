# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 03:27:28 2020
Last update on 30/10/2020

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Fill missing values of AOD (Aqua or Terra) according to regression equations 
(AOD_AT_reg.py)
Only changes based on the band used for AOD calculation
Python2.7 (env: aq-py27)

"""
# parameters

band = "55" # or 55
ini = 2013
end = 2019



import os
import glob
import numpy as np
from pyhdf.SD import SD, SDC
import json




if band  == "55":
    DATAFIELD_NAME = 'Optical_Depth_055'
else:
    DATAFIELD_NAME = 'Optical_Depth_047'
print ("band for AOD retrieval: ", DATAFIELD_NAME)



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
    # Q3D = hdf.select("AOD_QA")
    # Read attributes.
    attrs = data3D.attributes(full=1)
    fva=attrs["_FillValue"]
    _FillValue = fva[0]
    sfa=attrs["scale_factor"]
    scale_factor = sfa[0]
    return data3D, _FillValue, scale_factor, orb, orb_time


for year in range (ini, end+1):
    print ("AOD for " + str(year) + "\n")
    Name = "AOD" + str(year-2000)
    directory = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD' , Name, "AOD" + str(year))
    # Opening JSON file 
    nameAT = directory + "_AT_reg_" + "AOD_" + band + ".json"   
    nameTA = directory + "_TA_reg_" + "AOD_" + band + ".json"  
    pathAT = os.path.join(directory, nameAT)
    f = open(pathAT,)
    # returns JSON object as  # a dictionary 
    dicAT = json.load(f) 
    b1 = dicAT['coefAT']
    b0 = dicAT['intcAT']
    pathTA = os.path.join(directory, nameTA)
    f = open(pathTA,)
    # returns JSON object as  # a dictionary 
    dicTA = json.load(f) 
    a1 = dicTA['coefTA']
    a0 = dicTA['intcTA']
    for file in glob.glob(directory + '\*.hdf'):
        FILEPATH = file
        print(FILEPATH)
        data3D, _FillValue, scale_factor, orb, orb_time = read_dataset(DATAFIELD_NAME, FILEPATH)
        dataT = []
        dataA = []
        aod = np.zeros((1200, 1200))
        # i=1
        for k in range (orb):
             plat = orb_time[k][-1]
             data = data3D[k,:,:].astype(float)
             data[data==_FillValue] = np.nan
             if plat == "T":
                 dataT.append(data)
             else: 
                 dataA.append(data)
             # i+=1
                 
        meanT = np.nanmean(dataT, axis=0)
        meanA = np.nanmean(dataA, axis=0)
        maskT = np.isnan(meanT)
        maskA = np.isnan(meanA)
        if len(dataA) == 0:
            maskA = np.ones((1200, 1200), dtype = bool)
            meanA = np.empty((1200, 1200))
        if len(dataT) == 0:
            maskT = np.ones((1200, 1200), dtype = bool)
            meanT = np.empty((1200, 1200))
        for i in range (1200):
            for j in range(1200):
                # print(i,j)
                if (maskA[i,j] == True) and (maskT[i,j] == False):
                    # TA
                    meanA[i,j] = a1* meanT[i,j] + a0
                    aod[i,j] = int(round(0.5*(meanA[i,j]+meanT[i,j]))) 
                    
                elif (maskT[i,j] == True) and (maskA[i,j] == False):
                    # AT
                    meanT[i,j]  = b1* meanA[i,j]  + b0
                    aod[i,j] = int(round(0.5*(meanA[i,j]+meanT[i,j])))
                    
                elif (maskT[i,j] == False) and (maskA[i,j] == False):
                    aod[i,j] = int(round(0.5*(meanA[i,j]+meanT[i,j])))
                      
                else:
                    aod[i,j] = int(-28672) 
                    
                
        output_name = "AOD_"+orb_time[0][0:7]+"_"+band+".npz"
        outputpath = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD',"AOD"+str(year-2000), output_name)
        # np.save(output, aod)
        np.savez_compressed(outputpath, a=aod)
        
        






# print('\nAll valid files have been processed')    

