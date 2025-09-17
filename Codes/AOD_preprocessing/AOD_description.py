# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 03:26:46 2020

@author: Hossein Bagheri
"""
import os
import gdal

path = os.path.join("G:\project\AQ\data-Iran_Tehran",
                    "AOD19",
                    "MCD19A2.A2019274.h22v05.006.2019276040344.hdf")

#Lists all subdatasets of any one file
file = gdal.Open(path)
for path, desc in file.GetSubDatasets():
    print(desc)




