# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:36:49 2021

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Mask generation for Tehran city
"""

City = "Tehran"
px = [500, 530]
py = [169, 238]
import os
root = "G:\project\AQ\data-Iran_Tehran"
path_shx = os.path.join(root, "QGIS-project\Tehran_BB\Tehran_ter.shp")
import numpy as np

import fiona
shape = fiona.open(path_shx)
print(shape.schema)

#first feature of the shapefile
first = shape.next()
print(first)# (GeoJSON format)

from shapely.geometry import shape
shp_geom = shape(first['geometry']) # or shp_geom = shape(first) with PyShp)
# print(shp_geom)

# path of latitude and longtitude files
PATH_LAT = os.path.join("G:\project\AQ",
                    "data-Iran_Tehran",
                    "lat.txt")
PATH_LONG = os.path.join("G:\project\AQ",
                    "data-Iran_Tehran",
                    "lon.txt")

latitude = np.loadtxt(PATH_LAT)
longtitude = np.loadtxt(PATH_LONG)
mask= np.zeros((px[1]-px[0], py[1]-py[0]), dtype=bool)
from shapely.geometry import Point 
for i in range(px[0],px[1]):
    for j in range(py[0],py[1]):
         point = Point(longtitude[i,j], latitude[i,j])
         mask[i-px[0],j-py[0]] = shp_geom.contains(point)
         
output_name = "Tehran_mask.npz"
outputpath = os.path.join(root, output_name)
        # np.save(output, aod)
np.savez_compressed(outputpath, m=mask)


         
         

