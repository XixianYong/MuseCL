import csv

import numpy as np
from tqdm import tqdm
from mobility import which_region

grid = np.load("").tolist()
region_idx = np.load("").tolist()

POI_loc_num = dict()

file_path_ = ""
poi_name_last = ''
for i in tqdm(['5', '1', '2', '3', '4']):
    file_path = file_path_ + i + '.csv'
    csv_reader = csv.reader(open(file_path))
    for line in tqdm(csv_reader):
        if line[3] == 'poi-name':
            pass
        else:
            if line[3] == poi_name_last:
                pass
            else:
                poi_name_last = line[3]
                lon = float(line[6])
                lat = float(line[7])
                region_loc = which_region(lon, lat, grid)
                region_num = str(region_loc[0]*53 + region_loc[1])
                if region_num in region_idx:
                    if region_num in POI_loc_num:
                        POI_loc_num[region_num] = POI_loc_num[region_num] + 1
                    else:
                        POI_loc_num[region_num] = 1

print(POI_loc_num)

POI_num = []

for idx in region_idx:
    POI_num.append(POI_loc_num[idx])

print(POI_num)

np.save('BJ_POI_num.npy', np.array(POI_num))
