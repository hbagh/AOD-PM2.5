"""
Created on Thu Jan  7 16:26:52 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Yearly AOD calculation
env: Python3
"""
City = "Tehran"
ini = 2013
end = 2019
_FillValue = -28672

import glob
import datetime
import numpy as np
import os
root = 'G:\project\AQ\data-Iran_Tehran'


def jdtodatestd (jdate):
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)
mask_path = os.path.join(root, "Tehran_mask.npz")
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]
List = []
path = os.path.join(root, 'Grid\AOD\daily_AOD')
for file in glob.glob(path + '\*.npz'):
    date=jdtodatestd(file[-11:-4])
    List.append((date.year, date.month, file))

for year in range (ini, end+1):
    yList = [t[0:3] for t in List if t[0] == year]
    print ("Year:", year)
    aod_list = []
    for i in range(len(yList)):
        loaded_aod = np.load(yList[i][2])
        aod = loaded_aod["ac"]
        aod[aod == _FillValue] = np.nan
        aod_list.append(aod)
    aodm  = np.nanmedian(aod_list, axis=0)
    # aodm_masked = aodm[mask==False]= -9999
        
    output_name = "yearly_AOD_"+str(year)
    output = os.path.join(root, 'Grid\AOD\yearly_AOD', output_name)
    np.savez_compressed(output, ac=aodm)
    print(output_name+" was processed and generated")


