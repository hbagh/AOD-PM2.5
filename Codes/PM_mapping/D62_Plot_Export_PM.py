# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:26:52 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Plot PM2.5 on a map
env: Python3
"""

Date = 2018056
import geopandas as gpd
import os
import numpy as np
import gdal, osr
from scipy.ndimage import gaussian_filter
folder = 'Grid\PM\daily_int'
root = 'G:\project\AQ0\Data' 
shx_path = os.path.join(root, 'layers\limit\Tehran_limit.shp') 

mask_path = os.path.join(root, "Tehran_mask.npz")
top_left_lat = 35.8305 
top_left_lon = 51.0890
bottom_right_lat = 35.5632
bottom_right_lon = 51.6084
    
name = "daily_int_" + str(Date)  + '.npz'    
data_path = os.path.join(root, folder, name)
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]
loaded_aod = np.load(data_path)
pm = loaded_aod["pm"]

pmf = gaussian_filter(pm, sigma=3)

aoi_boundary = gpd.read_file(shx_path)
ax1 = aoi_boundary.plot(edgecolor="black", facecolor="None")
    
ax = (bottom_right_lon - top_left_lon)/(pm.shape[1])
ay = (top_left_lat - bottom_right_lat)/(pm.shape[0])
# extent_mat = (top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat)
extent_mat = (top_left_lon, top_left_lon + pm.shape[1] * ax, top_left_lat - pm.shape[0] * ay, top_left_lat)

ax1.autoscale(False)

ax1.imshow(pmf, extent=extent_mat, cmap='jet')

PM = []
name = "PM_int_" + str(Date)
dst_filename = os.path.join(root, folder, name+'.tiff')
    # # set geotransform
nx = pm.shape[1]
ny = pm.shape[0]

xmin, ymax = [top_left_lon, top_left_lat]
xres = ax
yres = ay
geotransform = ([xmin, xres, 0, ymax, 0, -yres])

# create the 3-band raster file
bands = 1 # number of bands
dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, nx, ny, bands, gdal.GDT_Float64)

dst_ds.SetGeoTransform(geotransform)    # specify coords
srs = osr.SpatialReference()            # establish encoding
srs.ImportFromEPSG(4326)                # WGS84 lat/long
dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
dst_ds.GetRasterBand(1).WriteArray(pmf)   # write r-band to the raster
# dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
# dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster
dst_ds.FlushCache()                     # write to disk
dst_ds = None
    
        
        

        





