import numpy as np

POI_count = np.load("")
region_idx = np.load("").tolist()

POI_num = []
for region in region_idx:
    region = int(region)
    x = region // 53
    y = region % 53
    num = POI_count[x][y]
    POI_num.append(num)

print(POI_num)
np.save("POI_num.npy", np.array(POI_num))
