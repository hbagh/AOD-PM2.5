# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:30:01 2020
Last update on 29/10/2020

@author: Hossein Bagheri
Gmail: h.bagheri.en@gmail.com
Description: 
Convert hourly air pollution measurments to a dataframe (from Excel imputs)
Additions: Convert local time to UTC, adding positions and ids of ground stations  
Python2.7 (env: base)
"""
# Parameters



ref_year = 2000
City = "Asia/Tehran"
LIST = ['Aqdasiyeh', 'Darrous', 'District10', 'District11', 'District16',
        'District19', 'District2', 'District21', 'District4', 'Fath.Square',
        'Golbarg', 'Mahallati', 'Masoudieh', 'Piroozi', 'Punak', 'Ray', 'District22',
        'Rose.Park', 'Sadr', 'Setad.bohran', 'ShadAbad', 'Sharif.University',
        'Tarbiat.Modares.University'] 


# libraries
import os
import pandas as pd
import datetime

# path to files
file_path = os.path.join("C:\Project\AQ",
                            "Data",
                            "stations"+".xlsx")

df = pd.DataFrame(pd.read_excel(file_path)) 



# def str2timestamp(date_string):
#     return datetime.datetime.strptime(date_string, "%Y-%m-%d")

# def getIndexes(dfObj, string):
#     ''' Get index positions of value in dataframe i.e. dfObj.'''
#     listOfPos = list()
#     # Get bool dataframe with True at positions where the given value exists
#     result = dfObj.isin([string])
#     # Get list of columns that contains the value
#     seriesObj = result.any()
#     columnNames = list(seriesObj[seriesObj == True].index)
#     # Iterate over list of columns and fetch the rows indexes where value exists
#     for col in columnNames:
#         rows = list(result[col][result[col] == True].index)
#         for row in rows:
#             listOfPos.append((row, col))
#     # Return a list of tuples indicating the positions of value in the dataframe
#     return listOfPos

# Dir = "daily13-19"
# for var in LIST:
#     print(var)
    

#     path = os.path.join("G:\project\AQ",
#                         "Data\Ground_measurements",
#                         Dir,
#                         var+".xlsx")
#     fileNameOut = var + ".csv"
#     out_path = os.path.join("C:\Project\AQ0",
#                             "Data\Ground_measurements",
#                             Dir,
#                             fileNameOut)
#     if os.path.exists(path) == False:
#         continue
#     # read an excel file and convert  
#     # into a dataframe object 
#     df = pd.DataFrame(pd.read_excel(path)) 


    

#     print(df.columns)
#     print(df.dtypes)

#     df['date'] = [str(str2timestamp(d).date()) for d in df['Date']]       
    
    

#     df = df.drop(["Date", "O3 ppb", "CO ppm", "NO ppb", "NO2 ppb", "NOx ppb", "SO2 ppb", 
#                   "PM 10 ug/m3"], axis = 1)

#     pos = getIndexes(df_station, var)[0][0]
#     idx = df_station.iloc[pos]["id"]
#     Lat = df_station.iloc[pos]["Latitude"]
#     Long = df_station.iloc[pos]["Longitude"]
#     df['station'] = pd.Series([idx for x in range(len(df.index)+1)])
#     df['lat'] = pd.Series([Lat for x in range(len(df.index)+1)])
#     df['long'] = pd.Series([Long for x in range(len(df.index)+1)])
#     df.to_csv(out_path, index=False)

# print("\n All stations were processed")

















