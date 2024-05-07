import csv
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def load_fence(filepath):
    buffer = []
    with open(filepath, 'rt', encoding='utf-8-sig') as vsvfile:
        reader = csv.reader(vsvfile)
        for row in reader:
            buffer = buffer + (list(map(float, row)))
    buffer = np.array(buffer).reshape(len(buffer) // 2, 2)
    return buffer


def grid_split():
    longitude_base = 115.4194
    latitude_base = 41.0647
    longitude_step = 0.04062224
    latitude_step = 0.008993

    region_loc = []

    filepath = ""
    fence = load_fence(filepath)
    count = 0

    for i in range(182):
        temp = []
        for j in range(53):
            if i % 2 == 0:
                start = longitude_base
            else:
                start = longitude_base + longitude_step / 2
            longitude = start + j * longitude_step
            latitude = latitude_base - i * latitude_step
            point = Point(longitude, latitude)
            if Polygon(fence).contains(point):
                count = count + 1
                loc = [longitude, latitude]
                temp.append(loc)
            else:
                loc = [0, 0]
                # loc = [longitude, latitude]
                temp.append(loc)
        region_loc.append(temp)
    return count, region_loc


a, b = grid_split()
