import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

from grid_division import load_fence
from mobility import which_region


# def get_index(idx):
#     loc_x = int(idx) // 30
#     loc_y = int(idx) % 30
#     return loc_x, loc_y
#
#
# def get_distance(lon_lat_list, idx_1, idx_2):
#     loc1_x, loc1_y = get_index(idx_1)
#     loc2_x, loc2_y = get_index(idx_2)
#     loc1_lon_lat = lon_lat_list[loc1_x][loc1_y]
#     loc2_lon_lat = lon_lat_list[loc2_x][loc2_y]
#
#     lon_distance = 85.030933 * abs(loc1_lon_lat[0] - loc2_lon_lat[0])
#     lat_distance = 111 * abs(loc1_lon_lat[1] - loc2_lon_lat[1])
#
#     return (lon_distance * lon_distance + lat_distance * lat_distance) ** 0.5


# region_idx = np.load("").tolist()
# grid = np.load("")
#
# popularity_density = dict()
# for region in region_idx:
#     popularity_density[region] = []
#
# data_path = ""
#
# shanghai = gpd.read_file("")
#
# data = csv.reader(open(data_path))
# for line in tqdm(data):
#     if line[1] != 'Y':
#         lon = float(line[0])
#         lat = float(line[1])
#         pd = float(line[2])
#         point = Point(lon, lat)
#         if point.within(shanghai.geometry.iloc[0]):
#             location = which_region(lon, lat, grid)
#             if str(30*location[0]+location[1]) in region_idx:
#                 popularity_density[str(30*location[0]+location[1])].append(pd)
#
# print(popularity_density)
# np.save('SH_pd_dict.npy', np.array(popularity_density))

pd_list = np.load("", allow_pickle=True).tolist()
region_idx = np.load("").tolist()
region_idx.sort()
beijing_pd = []
for region in region_idx:
    a = pd_list[region]
    sum = 0
    num = len(a)
    if num == 0:
        beijing_pd.append(0)
    else:
        for i in a:
            sum = sum + i
        beijing_pd.append(sum / num)

print(beijing_pd)
print(len(beijing_pd))
np.save('SH_pd.npy', np.array(beijing_pd))