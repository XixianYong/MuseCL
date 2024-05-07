import heapq
import os
import csv

import numpy as np
from random import choice

from tqdm import tqdm

similarity = np.load("").tolist()
region_idx = np.load("").tolist()

file_path = ""

a = []

for num in tqdm(range(10000)):
    anc_region = choice(region_idx)
    anc_similarity = np.array(similarity[int(anc_region)])
    top_simi = heapq.nlargest(2, range(len(anc_similarity)), anc_similarity.take)

    anc_image_list = [f for f in os.listdir(os.path.join(file_path, anc_region)) if not f.startswith('.')]
    anc_image_name = choice(anc_image_list)
    anc_path = os.path.join(anc_region, anc_image_name)
    # print(anc_path)

    pos_region = str(choice(top_simi[1:]))
    # print(similarity[int(anc_region)][int(pos_region)])
    pos_image_list = [f for f in os.listdir(os.path.join(file_path, pos_region)) if not f.startswith('.')]
    pos_image_name = choice(pos_image_list)
    pos_path = os.path.join(pos_region, pos_image_name)
    # print(pos_path)

    neg_region = choice(region_idx)
    while neg_region == anc_region or neg_region == pos_region:
        neg_region = choice(region_idx)
    neg_image_list = [f for f in os.listdir(os.path.join(file_path, neg_region)) if not f.startswith('.')]
    neg_image_name = choice(neg_image_list)
    neg_path = os.path.join(neg_region, neg_image_name)
    # print(neg_path)

    line = [anc_path, pos_path, neg_path]
    a.append(line)
print(a)

with open("", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(a)
