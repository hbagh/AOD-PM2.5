"""
Created on Sun Nov 15 09:32:35 2020

Last update on 30/10/2020

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
AOD quality extploitation from MAIAC products
Only changes based on the band used for AOD calculation
Python2.7 (env: aq-py27)

"""
# parameters

band = "47" # or 55 
ini = 2013
end = 2019
quality = "best"


import os
import glob
import numpy as np
from pyhdf.SD import SD, SDC



if band  == "55":
    DATAFIELD_NAME = 'Optical_Depth_055'
else:
    DATAFIELD_NAME = 'Optical_Depth_047'
print ("band for AOD retrieval: ", DATAFIELD_NAME)

# quality flag
def QAOD (val):
    string = '{0:016b}'.format(int(val))
    # print(string)
    CloudM = string[13:16]
    AdjM = string[8:11]
    QA = string[4:8]
    return CloudM, AdjM, QA

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
    Q3D = hdf.select("AOD_QA")

    return Q3D, orb, orb_time

  
for year in range (ini, end+1):
    print ("Quality for " + str(year) + "\n")
    Name = "AOD" + str(year-2000)
    directory = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD' , Name, "AOD" + str(year))
    AOD_station = []
    AOD_station_new = []
    
    for file in glob.glob(directory + '\*.hdf'):
        FILEPATH = file
        print(FILEPATH)
        Q3D, orb, orb_time = read_dataset(DATAFIELD_NAME, FILEPATH)
  
        Qt = []
        # A =[]
        for k in range (orb):
            Qaod= Q3D[k,:,:]
            # A.append(Qaod[0:1, 276:277])
            for i in range(0,1200):
                for j in range (0,1200):
                    # print(Qaod[i,j])
                    CloudM, AdjM, QA = QAOD(Qaod[i,j])
                    # print(CloudM, AdjM, QA)
                    Qaod[i,j]=0
                    # print(Qaod[i,j])
                    # if  CloudM == "001" and AdjM == "000" and QA == "0000":
                    #     Qaod[i,j] = 1 
                    if CloudM == "001" or CloudM == "010"  and AdjM == "000":
                        Qaod[i,j] = 1 
                    # print(Qaod[i,j])
            Qt.append(Qaod)
        Qtprob = np.mean(Qt, axis = 0)

        outname = "QProb_"+quality+orb_time[0][0:7]+"_"+band+".npz"
        outpath = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD',"AOD"+str(year-2000), outname)
        # np.save(outpath, Qtprob)
        np.savez_compressed(outpath, qp=Qtprob)


print('\nAll valid files have been processed')    




