"""
Meteorological data 
convert NetCDF4 to csv data 
Python: 2.7
"""
ini = 2013
end = 2019
name = "met"


import xarray as xr
import os
import math

# wind spped 
def WS(u, v):
    return math.sqrt(u**2 + v**2)
# wind direction
def WD(x, y):
    return math.atan2(y, x)
    

for year in range (ini ,end+1):
    met_fileName = str(year) + "_" + name + ".nc"
    file_location = os.path.join("G:\project\AQ",
                        "data-Iran_Tehran\MET", met_fileName)
    
    csv_file_out = str(year) + "_" + name + ".csv"
    out_directory = 'G:\project\AQ\data-Iran_Tehran\MET'
    out_path= os.path.join(out_directory, csv_file_out)
    
    
    ds = xr.open_dataset(file_location)
    df = ds.to_dataframe()
    
    
    df["ws10"] = df.apply(lambda row: WS(row['u10'], row['v10']), axis=1)
    df["wd10"] = df.apply(lambda row: WD(row['u10'], row['v10']), axis=1)

    df.to_csv(out_path)
    print(str(year)+ " data was processed")



