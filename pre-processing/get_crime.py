import csv
import json
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import folium

city = ox.geocode_to_gdf("New York")
boundary_polygon = city.geometry.iloc[0]


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = [json.loads(line) for line in file]
    return json_data


def distance(pos_1, pos_2):
    return pow(pow(pos_1[0] - pos_2[0], 2) + pow(pos_1[1] - pos_2[1], 2), 0.5)


def which_region(longitude, latitude, grid):
    longitude_base = -74.258843
    latitude_base = 40.91763
    longitude_step = 0.04062224 / 2
    latitude_step = 0.008993 / 2

    diff_lat = latitude_base - latitude
    x_flag = int(diff_lat / latitude_step)

    diff_lon = longitude - longitude_base
    y_flag = int(diff_lon / (longitude_step / 2))

    temp_1 = 0
    temp_2 = 0
    if (x_flag + y_flag) % 2 == 0:
        if y_flag % 2 == 0:
            temp_1 = distance([longitude, latitude], grid[x_flag][int(y_flag / 2)])
            temp_2 = distance([longitude, latitude], grid[x_flag + 1][int(y_flag / 2)])
            x = x_flag if temp_1 < temp_2 else x_flag + 1
            y = int(y_flag / 2)
        else:
            temp_1 = distance([longitude, latitude], grid[x_flag][int((y_flag - 1) / 2)])
            temp_2 = distance([longitude, latitude], grid[x_flag + 1][int((y_flag + 1) / 2)])
            x = x_flag if temp_1 < temp_2 else x_flag + 1
            y = int((y_flag - 1) / 2) if temp_1 < temp_2 else int((y_flag + 1) / 2)
    else:
        if y_flag % 2 == 0:
            temp_1 = distance([longitude, latitude], grid[x_flag][int(y_flag / 2)])
            temp_2 = distance([longitude, latitude], grid[x_flag + 1][int(y_flag / 2)])
            x = x_flag if temp_1 < temp_2 else x_flag + 1
            y = int(y_flag / 2)
        else:
            temp_1 = distance([longitude, latitude], grid[x_flag][int((y_flag + 1) / 2)])
            temp_2 = distance([longitude, latitude], grid[x_flag + 1][int((y_flag - 1) / 2)])
            x = x_flag if temp_1 < temp_2 else x_flag + 1
            y = int((y_flag + 1) / 2) if temp_1 < temp_2 else int((y_flag - 1) / 2)

    location = [x, y]
    return location


grid_split = np.load("").tolist()
reader = csv.reader(open(""))
crime = [[0 for _ in range(30)] for _ in range(100)]

for line in reader:
    if line[0] == 'INCIDENT_KEY' or line[19] == '':
        pass
    else:
        lon = float(line[19])
        lat = float(line[18])
        point = Point(lon, lat)
        if boundary_polygon.contains(point):
            [x, y] = which_region(lon, lat, grid_split)
            crime[x][y] = crime[x][y] + 1

region_idx = np.load("").tolist()
crime_num = []
for re in region_idx:
    x = re // 30
    y = re % 30
    crime_num.append(crime[x][y])

import math

crime_num_ln = []
for num in crime_num:
    crime_num_ln.append(math.log(num + 1))

print(crime_num)
print(crime_num_ln)
