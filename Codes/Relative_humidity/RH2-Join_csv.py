# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:43:47 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Concatenate all RH data-frames 
"""


import glob
import pandas as pd
import os


Name_out = "insitu_all" + ".csv"

root = "C:\Project\AQ0\Data"
data_path = "Ground_measurements\synoptic_data"


path = os.path.join(root, data_path)

 
out_path = os.path.join(path, Name_out)

i= 0

for file in glob.glob(path + '\*.csv'):
    if i == 0: 
         df0 = pd.read_csv(file)
         i +=1
         print (file + " " + "shape:", df0.shape)

    else:
        df = pd.read_csv(file)
        frame = pd.concat([df0, df], axis = 0)
        i +=1
        df0 = frame
        print (file + " " + "shape:", df.shape)

      
df0.to_csv(out_path , index=False)

df0['year'] = [y[-4:] for y in df0['new_date']]

for i in df0['year'].unique():
    df_slice = df0.loc[df0['year'] == i]
    outpath = os.path.join(root, data_path, i+'.csv')
    df_slice.to_csv(outpath, index = False)
    
    
