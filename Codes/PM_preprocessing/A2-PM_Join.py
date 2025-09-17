
"""
Created on Wed Oct 14 09:45:14 2020
Last update on 29/10/2020

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Concatenate all PM data-frames 
Phyton3 (env: base)
"""

year = 2013

ref_year = 2000

import glob
import pandas as pd
import os

Name_out = "PM" + ".csv"
Dir = "daily13-19"

path = os.path.join("G:\project\AQ",
                    "Data\Ground_measurements", 
                    Dir)

 
out_path = os.path.join("G:\project\AQ",
                        "Data\Ground_measurements",
                        Name_out)

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
    
    
