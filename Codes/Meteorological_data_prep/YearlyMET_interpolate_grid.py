# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:24:31 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Yearly Meteorological data calculation
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
    met_list = []
    for i in range(len(yList)):
        loaded_met = np.load(yList[i][2])
        met = loaded_met["met"]
            
        met_list.append(met)
    metm  = np.nanmean(met_list, axis=0)

    out_folder = "yearly_" + var
    out_path = os.path.join(root, "Grid\MET", out_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    output_name = "yearly_" + var + "_" + str(year)
    output = os.path.join(root, "Grid\MET", out_folder, output_name)
    np.savez_compressed(output, met=metm)
    print(output_name+" was processed and generated")
        
