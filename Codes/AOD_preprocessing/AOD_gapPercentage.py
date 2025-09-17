# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:05:07 2020
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
gap of data using different window sizes
Python 3, (env: base)
"""
# parameters
grid_size = 15

band = "55"


import pandas as pd
import os
from datetime import timedelta, date
import numpy as np
import datetime
import seaborn as sb
import matplotlib.pyplot as plt

root_path = "G:\project\AQ\data-Iran_Tehran"


nameAOD = "AOD"+"_g"+str(grid_size)+"_"+band+".csv" 
pathAOD = os.path.join(root_path,"AOD",nameAOD)


df = pd.read_csv(pathAOD)
percentage = len(df["AODm"].dropna())/(len(df.date.unique())*len(df.station.unique()))
print("percentage of gap of data: ", round(1-percentage,2))








