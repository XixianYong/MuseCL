import csv

import gensim
import numpy as np
from tqdm import tqdm

from mobility import which_region

model = gensim.models.Word2Vec.load('')
grid = np.load('grid_split.npy').tolist()
region_idx = np.load('region_idx.npy').tolist()
POI_rep = dict()

file_path_ = ""

for i in tqdm(['5', '1', '2', '3', '4']):
    file_path = file_path_ + i + '.csv'
    csv_reader = csv.reader(open(file_path))
    for line in tqdm(csv_reader):
        if line[3] == 'poi-name':
            pass
        else:
            lon = float(line[6])
            lat = float(line[7])
            idx = which_region(lon, lat, grid)
            if str(idx[0]*53 + idx[1]) in region_idx:
                flag = str(idx[0] * 53 + idx[1])
                if flag in POI_rep:
                    temp = line[5].split(';')
                    for item in temp:
                        POI_rep[flag].append(item)
                else:
                    POI_rep[flag] = []
                    temp = line[5].split(';')
                    for item in temp:
                        POI_rep[flag].append(item)

for key in POI_rep:
    POI_rep[key] = list(set(POI_rep[key]))

print(POI_rep)

for key in POI_rep:
    count = 0
    key_rep = np.zeros((256,), dtype=float)
    for word in POI_rep[key]:
        try:
            word_rep = model.wv[word]
            key_rep = key_rep + word_rep
            count = count + 1
        except KeyError:
            pass
    if count == 0:
        pass
    else:
        key_rep = key_rep / count

    POI_rep[key] = key_rep

np.save('', POI_rep)
print("DONE!")
