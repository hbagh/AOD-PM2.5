# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:51:12 2021

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
crop AOD file for the study area: Tehran
Python2.7 (env: aq-py27)

"""
import os
import numpy as np 
import glob
band = '55' 

# Upper-left, Lower-right of a City
# Tehran
city_position = np.array([[35.8305, 51.089],
                          [35.5632, 51.6084]])
ini = 2013
end = 2019
# City = "Tehran"

#calculation to find nearest point in data to entered location (haversine formula)
def find_position(user_lat, user_lon, latitude, longtitude):
    R=6371000 #radius of the earth in meters
    lat1=np.radians(user_lat)
    lat2=np.radians(latitude)
    delta_lat=np.radians(latitude-user_lat)
    delta_lon=np.radians(longtitude-user_lon)
    a=(np.sin(delta_lat/2))*(np.sin(delta_lat/2))+(np.cos(lat1))*(np.cos(lat2))*(np.sin(delta_lon/2))*(np.sin(delta_lon/2))
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d=R*c
    #gets (and then prints) the x,y location of the nearest point in data to entered location, accounting for no data values
    x,y = np.unravel_index(d.argmin(),d.shape)
    print('\nThe nearest pixel to your entered location is at: \nLatitude:',latitude[x,y],' Longitude:',longtitude[x,y])
    return x, y
    
    
    


# path of latitude and longtitude files
PATH_LAT = os.path.join("G:\project\AQ",
                    "data-Iran_Tehran",
                    "lat.txt")
PATH_LONG = os.path.join("G:\project\AQ",
                    "data-Iran_Tehran",
                    "lon.txt")

latitude = np.loadtxt(PATH_LAT)
longtitude = np.loadtxt(PATH_LONG)
location_x = []
location_y = []
idx = []
for i in range(2):
    x, y = find_position(city_position[i,0], city_position[i,1], latitude, longtitude)
    idx.append(i)
    location_x.append(x)
    location_y.append(y)

location_x = np.array(location_x)
location_y = np.array(location_y)
idx = np.array(idx)
position_tuple = (idx, city_position[:,0], city_position[:,1], location_x, location_y)
position = np.transpose(np.vstack(position_tuple))

for year in range (ini, end+1):
    print ("AOD for " + str(year) + "\n")
    Name = "AOD" + str(year-2000)
    root_dir = os.path.join('G:\project\AQ\data-Iran_Tehran\AOD' , Name)
    path_aod = os.path.join(root_dir, "aod"+band)
    path_Qbest = os.path.join(root_dir, "Qbest"+band)

    
    # AOD_station_new = []
    FILEPATH_aod = []
    for file in glob.glob(path_aod + '\*.npz'):
        FILEPATH_aod.append(file)
        # print(file)
    FILEPATH_Qb = []
    for file in glob.glob(path_Qbest + '\*.npz'):
        FILEPATH_Qb.append(file)
        


    
    for i in range(len(FILEPATH_aod)):
        print(FILEPATH_aod[i])
        date = FILEPATH_aod[i][-14:-7]
        loaded_aod = np.load(FILEPATH_aod[i])
        aod = loaded_aod["a"]        
        loaded_Qb = np.load(FILEPATH_Qb[i])
        Qb = loaded_Qb['qp']
        crop_aod = aod[int(position[0,3]):int(position[1,3]), int(position[0,4]):int(position[1,4])]
        crop_Qb = Qb[int(position[0,3]):int(position[1,3]), int(position[0,4]):int(position[1,4])]
        output_name = "AOD_"+date+".npz"
        out_path = 'G:\project\AQ1\Data\Grid\AOD\daily'
        if not os.path.exists(out_path):
                os.mkdir(out_path) 
        np.savez_compressed(os.path.join(out_path, output_name), ac=crop_aod)
        
        output_name = "Qb_"+date+".npz"
        np.savez_compressed(os.path.join(out_path, output_name), qp=crop_Qb)
  




