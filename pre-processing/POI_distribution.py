import csv
import os

import folium
import webbrowser
from folium.plugins import HeatMap
import numpy as np
from tqdm import tqdm

grid = np.load('').tolist()
grid.sort()
loc = np.load("").tolist()

gcj_x = []
gcj_y = []

ll = ""
csv_reader = csv.reader(open(ll))

fp = ""
fl = os.listdir(fp)
for name in fl:
    if name == '.DS_Store':
        pass
    else:
        file_path = os.path.join(fp, name)
        image_list = os.listdir(file_path)
        for i in tqdm(image_list):
            idx = i.split('_')[0]
            csv_reader = csv.reader(open(ll))
            for line in csv_reader:
                if line[0] == idx:
                    print(idx)
                    gcj_x.append(float(line[1]))
                    gcj_y.append(float(line[2]))
                    break

print(gcj_x)

LNG_new = gcj_y
LAT_new = gcj_x
ration = [1 for _ in range(len(LNG_new))]


# ration = []
# fp = ""
# fl = os.listdir(fp)
# fl.sort()
# for name in fl:
#     if name in grid:
#         path = os.path.join(fp, name)
#         num = len(os.listdir(path))
#         ration.append(num)

data1 = list(zip(LNG_new, LAT_new, ration))
Center = [np.mean(np.array(LAT_new, dtype='float32')), np.mean(np.array(LNG_new, dtype='float32'))]
m = folium.Map(location=Center, zoom_start=6, tiles='Stamen Terrain')
HeatMap(data1).add_to(m)

name = 'POI_distribution.html'
m.save(name)
