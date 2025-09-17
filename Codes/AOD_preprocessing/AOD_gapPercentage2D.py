# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:45:46 2021


Gmail: h.bagheri.en@gmail.com
Description: 
gap of data using different window sizes
Python 3, (env: base)
"""
# parameters

band = "55"
ini = 2013
end = 2019

import os
import glob
import numpy as np



gap_per = []
Date = []
for year in range (ini, end+1):
    print ("AOD for " + str(year) + "\n")
    Name = "AOD" + str(year-2000)
    root_dir = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD' , Name)
    path_aod = os.path.join(root_dir, "aod"+band)

    
    # AOD_station_new = []
    FILEPATH_aod = []
    for file in glob.glob(path_aod + '\*.npz'):
        FILEPATH_aod.append(file)
        # print(file)
        


    
    for i in range(len(FILEPATH_aod)):
        print(FILEPATH_aod[i])
        date = FILEPATH_aod[i][-14:-7]
        loaded_aod = np.load(FILEPATH_aod[i])
        aod = loaded_aod["a"]
        aod[aod == -28672] = np.nan
        gap_per.append(np.round(np.sum(np.isnan(aod))/(aod.shape[0]*aod.shape[1]),2))
        Date.append(date)   
        
       
import matplotlib.pyplot as plt
plt.figure
plt.hist(gap_per)   
plt.show()  





# output_name = "AOD"+"_g"+str(grid_size)+"_"+band+".csv"
# output = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD', output_name)
# # df.dropna(inplace=True)
# df.to_csv(output, index=False)

# print('\nAll valid files have been processed')    





