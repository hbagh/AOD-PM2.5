# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:07:06 2021
@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Daily PM Interpolation: IDW
env: python3
"""
#2018

root = 'G:\project\AQ0\Data'

import glob
import pandas as pd
import math
import numpy as np
import os

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

# # know points
# x = [-47.6, -48.9, -48.2, -48.9, -47.6, -48.6]
# y = [-23.4, -24.0, -23.9, -23.1, -22.7, -22.5]
# z = [27.0,  33.4,  34.6,  18.2,   30.8, 42.8]
# # unknow point
# xi = [-48.0530600]
# yi = [-23.5916700]
# # running the function
# idwr(x,y,z,xi,yi)
# # output
# [[-48.05306, -23.59167, 31.486682779040855]]



list_csv = []
for file in glob.glob(os.path.join(root, 'Grid\PM\daily') + '\*.csv'):
    list_csv.append(file)

list_npz = []
for file in glob.glob(os.path.join(root, 'Grid\PM\daily') + '\*.npz'):
    list_npz.append(file)
    

mask_path = os.path.join(root, "Tehran_mask.npz")
mask_loaded = np.load(mask_path)
mask = mask_loaded["m"]

top_left_lat = 35.8305 
top_left_lon = 51.0890
bottom_right_lat = 35.5632
bottom_right_lon = 51.6084 
# 51.0725, 51.6233, 35.5873, 35.8318   
grid_lat = np.linspace( top_left_lat, bottom_right_lat, mask.shape[0])
grid_lon = np.linspace(top_left_lon, bottom_right_lon, mask.shape[1])
    
long_data , lat_data = np.meshgrid(grid_lon, grid_lat)

for i in range(len(list_npz)):
    print(list_npz[i])
    Date = list_npz[i][-11:-4]
    loaded_pm = np.load(list_npz[i])
    pm = loaded_pm["pm"]
    
    
    df = pd.read_csv(list_csv[i])
    x = df["long"].tolist()
    y = df["lat"].tolist()
    z = df["PM"].tolist()
    
    idx = np.argwhere(pm==-9999)

    # for r in range(pm.shape[0]):
    #     for c in range(pm.shape[1]):
    #         if pm[r,c] == -9999:
    #             xi = [long_data[r,c]]
    #             yi = [lat_data[r,c]]
    #             
    #             pm[r,c] = 
    #         else: 
    #             continue
    for i in range(len(idx)):
        xi = [long_data[idx[i,0],idx[i,1]]]
        yi = [lat_data[idx[i,0],idx[i,1]]]
        val = idwr(x,y,z,xi,yi)  
        pm[idx[i,0],idx[i,1]] = val[0][2]
    export_path = os.path.join(root, 'Grid\PM\daily_int', "daily_int_"+Date+".npz")
    np.savez_compressed(export_path, pm=pm)

        

    # cos = layer.extent()
    # output = QgsGridFileWriter(IDW_interpolator,export_path,cos,2000,2000)
    # output.writeFile(True)
    

   
        
                
                
                
    
    
    
    
    
    
    
    
    
    
    
    
    # layer = QgsVectorLayer(file, "data", "delimitedtext")
    # if not layer.isValid():
    #     print("Layer not loaded")
    # Date=file[-11:-4]
    # layer_data = QgsInterpolator.LayerData()
    # layer_data.source = layer
    # layer_data.zCoordInterpolation = False
    # layer_data.interpolationAttribute = 0
    # layer_data.sourceType = QgsInterpolator.SourcePoints

    # IDW_interpolator = QgsIDWInterpolator([layer_data])
    # IDW_interpolator.setDistanceCoefficient(2)

    # export_path = os.path.join(root, 'PM\daily_jointPM', "daily_int_"+Date+".tif")
    # cos = layer.extent()
    # output = QgsGridFileWriter(IDW_interpolator,export_path,cos,2000,2000)
    # output.writeFile(True)

# rlayer = iface.addRasterLayer(export_path, "interpolation_output")

    






# def krig2D_param(data):
#     param_dict2d = {
#                 "method": ["ordinary","universal"],
#                 "variogram_model": ["linear", "power", "gaussian", "spherical"],
#                 "pseudo_inv_type":["pinvh"]}

#     train = data[:, 0:2]
#     val = data[:,2]

#     estimator = GridSearchCV(Krige(), param_dict2d, verbose=True, return_train_score=True, scoring = "neg_root_mean_squared_error")
#     estimator.fit(X=train, y=val)

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




 
     
# def krig2D(data, query, method, variogram):

#     val = data[:,2]
#     xtrain = data[:, 0]
#     ytrain = data[:, 1]

#     xtest = query[:, 0]
#     ytest = query[:, 1]
                
    
#     if method == 'universal':
#         UK = UniversalKriging(xtrain, ytrain, val,
#                               variogram_model=variogram,
#                               pseudo_inv_type="pinvh")
#         k,_ = UK.execute("points", xtest, ytest)
    
#     else:
#         OK = OrdinaryKriging(xtrain, ytrain, val,
#                              variogram_model=variogram,
#                              pseudo_inv_type="pinvh")
#         k,_ = OK.execute("points", xtest, ytest)
#     return k

# # path of latitude and longtitude files
# PATH_LAT = os.path.join(root,
#                     "lat.txt")
# PATH_LONG = os.path.join(root,
#                     "lon.txt")

# latitude = np.loadtxt(PATH_LAT)
# longtitude = np.loadtxt(PATH_LONG)


# def jdtodatestd (jdate):
#     fmt = '%Y%j'
#     datestd = datetime.datetime.strptime(jdate, fmt).date()
#     return(datestd)
# mask_path = os.path.join(root, "Tehran_mask.npz")
# mask_loaded = np.load(mask_path)
# mask = mask_loaded["m"]
# List = []
# List2 = []
# path = os.path.join(root, 'AOD\Tehran\AODv1')
# for file in glob.glob(path + '\*.npz'):
#     date=jdtodatestd(file[-22:-15])
#     List.append((date.year, date.month, date.day, file))
#     new_file = file.replace("v1", "v2")
#     List2.append((date.year, date.month, date.day, new_file))
 

# i = 0
# while i < len(List):
#     loaded_aod = np.load(List[i][3])
#     aod = loaded_aod["ac"]
#     print(List[i][3])
#         # aod[aod == _FillValue] = np.nan
#         # aod[mask==False]=np.nan
#     aod_masked = aod[mask==True]
#     aod_masked[aod_masked == _FillValue] = np.nan
#     aod_index = np.argwhere(mask==True)
#     q2idx = aod_index[np.argwhere(np.isnan(aod_masked))[:,0]]
#     d2idx = np.argwhere(aod != _FillValue)
#     data2_aod = np.zeros(d2idx.shape[0]).reshape(-1,1)
#     for m in range(d2idx.shape[0]):
#         x=d2idx[m,0]
#         y=d2idx[m,1]
#         data2_aod[m] = aod[x, y]
  
#     data2 = np.concatenate((d2idx.astype("float"), data2_aod), axis = 1)
#     # param = krig2D_param(data2)
#     aod_int = krig2D(data2, q2idx.astype("float"), "universal", "spherical")
#     aod[mask==False]=np.nan
#     for count, l in enumerate(q2idx):
#         aod[l[0], l[1]] = aod_int[count]
#     print(sum(np.isnan(aod[mask==True])))
#     output_name = List[i][3][-22:-4]
#     outputv2 = os.path.join(root, 'AOD\Tehran\AODv2', output_name)
#     np.savez_compressed(outputv2, ac=aod)
#     i+=1
#     # else:
#     #     print(List[i][3])
#     #     # loaded_aod = np.load(List2[i-2][3])
#     #     # aod_1 = loaded_aod["ac"]
#     #     loaded_aod = np.load(List2[i-1][3])
#     #     aod0 = loaded_aod["ac"]
#     #     loaded_aod = np.load(List[i][3])
#     #     aod = loaded_aod["ac"]
#     #     aod_masked = aod[mask==True]
#     #     aod_masked[aod_masked == _FillValue] = np.nan
#     #     aod_index = np.argwhere(mask==True)
#     #     q3idx = aod_index[np.argwhere(np.isnan(aod_masked))[:,0]]
        
#     #     aods = np.stack((aod0, aod))
#     #     d3idx = np.argwhere(aods != _FillValue)
#     #     data3_aod = np.zeros(d3idx.shape[0]).reshape(-1,1)
#     #     for m in range(d3idx.shape[0]):
#     #         t=d3idx[m,0]
#     #         x=d3idx[m,1]
#     #         y=d3idx[m,2]
#     #         data3_aod[m] = aods[t, x, y]
  
#     #     data3 = np.concatenate((d3idx.astype('float64'), data3_aod), axis = 1)
        
        
#     #     aod_int = krig3D(data3, q3idx.astype('float64'), "universal3d", "spherical")
        
#     #     for count, l in enumerate(q2idx):
#     #           aod[l[0], l[1]] = aod_int[count]
#     #     print(sum(np.isnan(aod[mask==True])))
#     #     output_name = List[i][3][-22:-1]
#     #     print(sum(np.isnan(aod[mask==True])))
#     #     output_name = List[i][3][-22:-4]
#     #     outputv2 = os.path.join(root, 'AOD\Tehran\AODv2', output_name)
#     #     np.savez_compressed(outputv2, ac=aod)
#     #     i+=1
        
            
        
        
        


        
        

        
    
        
    
                
            
            

            
        
    
        

        
        
        
        
        
    

    



