import csv
import numpy as np
from tqdm import tqdm

from mobility import which_region

region_list = np.load("").tolist()
grid = np.load("").tolist()

csv_reader = csv.reader(
        open(""))

hn = dict()
id = 0
for line in tqdm(csv_reader):
    if line[3] != 'title':
        if line[0] != id:
            id = line[0]
            lon = float(line[12])
            lat = float(line[13])
            loc = which_region(lon, lat, grid)
            region_idx = str(loc[0] * 30 + loc[1])

            if region_idx in region_list:
                if region_idx in hn:
                    hn[region_idx].append(line[1])
                else:
                    hn[region_idx] = []
                    hn[region_idx].append(line[1])

print(hn)
region_idx_shzz = []
for key in hn:
    region_idx_shzz.append(key)

region_idx_shzz.sort()
# np.save('SH_region_idx_shzz.npy', np.array(region_idx_shzz))
print(region_idx_shzz)
print(len(region_idx_shzz))

SH_hn = []
for key in region_idx_shzz:
    price = hn[key]
    sum = 0
    num = 0
    for item in price:
        item = item
        # sum = sum + item
        num = num + 1
    # avg_price = sum / num
    SH_hn.append(num)

print(SH_hn)
print(len(SH_hn))
np.save('SH_house_num.npy', np.array(SH_hn))
