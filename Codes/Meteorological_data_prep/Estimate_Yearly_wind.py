# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:49:08 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Yearly PM calculation
env: Python3
"""
City = "Tehran"
ini = 2013
end = 2019
_FillValue = -9999

import glob
import datetime
import numpy as np
import os
import math
root = 'G:\project\AQ\data-Iran_Tehran'



def jdtodatestd (jdate):
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)
mask_path = os.path.join(root, "Tehran_mask.npz")
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]

top_left_lat = 35.8318 
top_left_lon = 51.0725
bottom_right_lat = 35.5873
bottom_right_lon = 51.6233  
# 51.0725, 51.6233, 35.5873, 35.8318   
grid_lat = np.linspace( top_left_lat, bottom_right_lat, mask.shape[0])
grid_lon = np.linspace(top_left_lon, bottom_right_lon, mask.shape[1])
    
long_data , lat_data = np.meshgrid(grid_lon, grid_lat)

Listwd = []
path = os.path.join(root, 'Grid\MET\daily_'+"wd10")
for file in glob.glob(path + '\*.npz'):
    date=jdtodatestd(file[-11:-4])
    Listwd.append((date.year, date.month, file))
    
Listws = []    
path = os.path.join(root, 'Grid\MET\daily_'+"ws10")
for file in glob.glob(path + '\*.npz'):
    date=jdtodatestd(file[-11:-4])
    Listws.append((date.year, date.month, file))


per = []
for year in range (ini, end+1):
    yList = [t[0:3] for t in Listwd if t[0] == year]
    wd_list = []
    for i in range(len(yList)):
        loaded_wd = np.load(yList[i][2])
        wd = loaded_wd["met"]
        wd_list.append(wd)
    wdm  = np.median(wd_list, axis=0)
    wdm = wdm * (180/math.pi)*(-1) + 90
    
    yList = [t[0:3] for t in Listws if t[0] == year]
    ws_list = []
    for i in range(len(yList)):
        loaded_ws = np.load(yList[i][2])
        ws = loaded_ws["met"]
        ws_list.append(ws)
    wsm  = np.median(ws_list, axis=0)


    if not os.path.exists(os.path.join(root, 'Grid\MET')):
        os.mkdir(os.path.join(root, 'Grid\MET'))
    # output_name = 'yearly_'+typePM + str(year)
    # output = os.path.join(root, 'Grid\MET\yearly_'+typePM, output_name+'.npz')
    # np.savez_compressed(output, met=wm)
    # print(output_name+" was processed and generated")
    
    ax = (bottom_right_lon - top_left_lon)/(wdm.shape[1])
    ay = (top_left_lat - bottom_right_lat)/(wdm.shape[0])
    W = []
    # Export PM2.5 as csv
    for c in range(wdm.shape[1]):
        Long = top_left_lon + ax * c
        for r in range(wdm.shape[0]):
            Lat = top_left_lat - ay * r
            W.append([Long, Lat, wdm[r,c], wsm[r,c]])
    
    import pandas as pd
    W = np.asarray(W)
    df = pd.DataFrame(data=W, columns=["Long", "Lat", "wind_direction", "wind_speed"])           
    output_name = 'yearly_' + "wind" + str(year) + ".csv"
    out_path = os.path.join(root, 'Grid\MET', output_name)
    df.to_csv (out_path, index = False)
    print(output_name + " generated")
    
    # import gdal, osr
    # dst_filename = os.path.join(root, 'Grid\MET\yearly_'+typePM, output_name+'.tiff')
    # # set geotransform
    # nx = wm.shape[1]
    # ny = wm.shape[0]

    # xmin, ymax = [top_left_lon, top_left_lat]
    # xres = (bottom_right_lon - top_left_lon)/(wm.shape[1])
    # yres = (top_left_lat - bottom_right_lat)/(wm.shape[0])
    # geotransform = ([xmin, xres, 0, ymax, 0, -yres])

    # # create the 3-band raster file
    # bands = 1 # number of bands
    # dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, nx, ny, bands, gdal.GDT_Float64)

    # dst_ds.SetGeoTransform(geotransform)    # specify coords
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(4326)                # WGS84 lat/long
    # dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    # dst_ds.GetRasterBand(1).WriteArray(wm)   # write r-band to the raster
    # # dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
    # # dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster
    # dst_ds.FlushCache()                     # write to disk
    # dst_ds = None
    
