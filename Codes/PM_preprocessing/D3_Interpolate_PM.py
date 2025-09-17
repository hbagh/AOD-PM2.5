# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:07:06 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Daily PM Interpolation: IDW
env: py-qgis
"""

Date = 2018056
root = 'G:\project\AQ0\Data'

from pykrige.rk import Krige
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import pandas as pd
import math
import numpy as np
import os
# import gdal, osr
folder = 'Grid\PM\daily'
# Distance calculation, degree to km (Haversine method)
def harvesine(lon1, lat1, lon2, lat2):
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return(d)
# ------------------------------------------------------------
# Prediction
def idwr(x, y, z, xi, yi):
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        xyzi = [xi[p], yi[p], u]
        lstxyzi.append(xyzi)
    return(lstxyzi)


     
def krig2D(x,y,z, method, variogram):
   
    if method == 'universal':
        interpolate = UniversalKriging(x, y, z,
                              variogram_model=variogram,
                              pseudo_inv_type="pinvh")

    
    elif method == 'ordinary':
        interpolate = OrdinaryKriging(x, y, z,
                             variogram_model=variogram,
                             pseudo_inv_type="pinvh")
        
    return interpolate

# def krig2D_param(x,y,z):
#     param_dict2d = {
#                 "method": ["ordinary","universal"],
#                 "variogram_model": ["linear", "power", "gaussian", "spherical"],
#                 "pseudo_inv_type":["pinvh"]}

#     train = np.vstack((np.asarray(x), np.asarray(y)))
#     train = train.T
    

#     estimator = GridSearchCV(Krige(), param_dict2d, verbose=True, return_train_score=True, scoring = "neg_root_mean_squared_error")
#     estimator.fit(X=train, y=z)

#     if hasattr(estimator, "best_score_"):
#         print("RMSE = {:.3f}".format(estimator.best_score_))
#         print("best_params = ", estimator.best_params_)
#         print("\nCV results::")
#     if hasattr(estimator, "cv_results_"):
#         for key in ["mean_test_score",
#                     "mean_train_score",
#                     "param_method",
#                     "param_variogram_model"
#                     ]:
#             print(" - {} : {}".format(key, estimator.cv_results_[key]))
    
#     return estimator.best_params_



mask_path = os.path.join(root, "Tehran_mask.npz")
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]

top_left_lat = 35.88305
top_left_lon = 51.0890
bottom_right_lat = 35.5632
bottom_right_lon = 51.6084    
  
grid_lat = np.linspace( top_left_lat, bottom_right_lat, mask.shape[0])
grid_lon = np.linspace(top_left_lon, bottom_right_lon, mask.shape[1])
    
long_data , lat_data = np.meshgrid(grid_lon, grid_lat)


file_name = "PM_est_" + str(Date)  + '.npz'
path = os.path.join(root, 'Grid\PM\daily', file_name)
if os.path.exists(path) == False:
    print(str(Date) + ' deos not exist')
else:
    loaded_pm = np.load(path)
    pm = loaded_pm['pm']
    
file_name = "PM_est_" + str(Date) + '.csv'
path = os.path.join(root, 'Grid\PM\daily', file_name)
if os.path.exists(path) == True:
    df_PMest = pd.read_csv(path)
    df1 = df_PMest.drop(columns = ['r', 'c'])
else:
    df1 = pd.DataFrame(columns = ['long', 'lat', 'PM'])
            
file_name = "PM_station_" + str(Date) + '.csv'
path = os.path.join('G:\project\AQ\Data', 'Ground_measurements\daily_export', file_name)
        
if os.path.exists(path) == True:
    df_PMground = pd.read_csv(path)
else:
    df_PMground = pd.DataFrame(columns = ['long', 'lat', 'PM'])
df = df1.append(df_PMground)
if df.shape[0] == 0:
    print(str(Date) + ' deos not exist')

else:
    df = df1.append(df_PMground)
    file_name = "PM_est_station_" + str(Date) + '.csv'
    out_path = os.path.join(root, 'Grid\PM\daily_est_station')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    df.to_csv(os.path.join(out_path, file_name) ,index = False)
    x = df["long"].tolist()
    y = df["lat"].tolist()
    z = df["PM"].tolist()
    interpolate = krig2D(x,y,z, 'universal', 'gaussian')
    idx = np.argwhere(pm==-9999)
    for i in range(len(idx)):
        #xi = [long_data[idx[i,0],idx[i,1]]]
        #yi = [lat_data[idx[i,0],idx[i,1]]]
        # val = idwr(x,y,z,xi,yi)
        pm[idx[i,0],idx[i,1]], _ = interpolate.execute("points", long_data[idx[i,0],idx[i,1]], lat_data[idx[i,0],idx[i,1]])        
        #pm[idx[i,0],idx[i,1]] = val[0][2]
    
            
    # shx_path = os.path.join(root, 'layers\limit\Tehran_limit.shp') 
    # mask_path = os.path.join(root, "Tehran_mask.npz")

    # import geopandas as gpd

    # aoi_boundary = gpd.read_file(shx_path)
    # ax1 = aoi_boundary.plot(edgecolor="black", facecolor="None")
    
    # ax = (bottom_right_lon - top_left_lon)/(pm.shape[1])
    # ay = (top_left_lat - bottom_right_lat)/(pm.shape[0])
    # extent_mat = (top_left_lon, top_left_lon + pm.shape[1] * ax, top_left_lat - pm.shape[0] * ay, top_left_lat)

    # ax1.autoscale(False)
    
    # # Finally, let's plot!
    # ax1.imshow(pm, extent=extent_mat, cmap='jet')
    # PM = []
    # name = "PM_est_" + str(Date)
    # dst_filename = os.path.join(root, folder, name+'.tiff')
    # # set geotransform
    # nx = pm.shape[1]
    # ny = pm.shape[0]

    # xmin, ymax = [top_left_lon, top_left_lat]
    # xres = ax
    # yres = ay
    # geotransform = ([xmin, xres, 0, ymax, 0, -yres])

    # # create the 3-band raster file
    # bands = 1 # number of bands
    # dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, nx, ny, bands, gdal.GDT_Float64)

    # dst_ds.SetGeoTransform(geotransform)    # specify coords
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(4326)                # WGS84 lat/long
    # dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    # dst_ds.GetRasterBand(1).WriteArray(pm)   # write r-band to the raster
    # # dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
    # # dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster
    # dst_ds.FlushCache()                     # write to disk
    # dst_ds = None
            
            
export_path = os.path.join(root, 'Grid\PM\daily_int', "daily_int_"+str(Date)+".npz")
print(export_path)
np.savez_compressed(export_path, pm=pm)
            
