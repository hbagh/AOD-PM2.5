# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:47:27 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Monty Meteorological data calculation
env: Python3
"""

ini = 2013
end = 2019
#LIST = ['d2m', 't2m', 'blh', 'lai_hv', 'lai_lv', 'sp', 'ws10', 'wd10', cdir, uvb]
var  = "uvb"
folder  = "daily_" + var
import glob
import datetime
import numpy as np
import os
root = 'G:\project\AQ\data-Iran_Tehran'


def jdtodatestd (jdate):
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)

List = []
path = os.path.join(root, 'Grid\MET', folder)
for file in glob.glob(path + '\*.npz'):
    date=jdtodatestd(file[-11:-4])
    List.append((date.year, date.month, file))
# per = []
for year in range (ini, end+1):
    yList = [t[0:3] for t in List if t[0] == year]
    print ("Year:", year)
    for month in range (1,13):
        mlist = [t[0:3] for t in yList if t[1] == month]
        # print(len(mlist))
        met_list = []
        for i in range(len(mlist)):
            loaded_met = np.load(mlist[i][2])
            met = loaded_met["met"]
            
            met_list.append(met)
        metm  = np.nanmean(met_list, axis=0)
        # percent = np.round(np.sum(np.isnan(aodm)/(aodm.shape[0]*aodm.shape[1])),2)
        # per.append(percent)
        # print("Year:{}, Month:{}, Gap percent: {}".format(year, month, percent))
        # aodm[np.isnan(aodm)]= -9999
        if month < 10: 
            smonth = "0"+str(month)
        else:
            smonth = str(month) 
        out_folder = "montly_" + var
        out_path = os.path.join(root, "Grid\MET", out_folder)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        output_name = "montly_" + var + "_" + smonth + "_" + str(year)
        output = os.path.join(root, "Grid\MET", out_folder, output_name)
        np.savez_compressed(output, met=metm)
        print(output_name+" was processed and generated")
        


# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(per, bins = 5 , color = "navy")
# plt.xlabel('Percentage of Missing AODs')
# plt.ylabel('Frequency')
# plt.show()







            
        
    

    


