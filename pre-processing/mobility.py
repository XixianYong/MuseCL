import os.path
import numpy as np
from tqdm import tqdm


def distance(pos_1, pos_2):
    return pow(pow(pos_1[0] - pos_2[0], 2) + pow(pos_1[1] - pos_2[1], 2), 0.5)


def which_region(longitude, latitude, grid):
    longitude_base = 115.4194
    latitude_base = 41.0647

    longitude_step = 0.04062224
    latitude_step = 0.008993

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


def mobility_count(root_path, mobility, grid):
    for num in tqdm(range(1, 10358)):
        file_path = os.path.join(root_path, str(num) + '.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        x_last = -1
        y_last = -1
        for line in lines:
            longitude = float(line.split(',')[2])
            latitude = float(line.split(',')[3][0:-1])
            if longitude < 115.4194 or longitude > 117.5103 or latitude < 39.4370 or latitude > 41.0647:
                pass
            else:
                loc = which_region(longitude, latitude, grid)
                if loc[0] != x_last or loc[1] != y_last:
                    if x_last == -1 and y_last == -1:
                        pass
                    else:
                        mobility[x_last][y_last][1] = mobility[x_last][y_last][1] + 1
                    mobility[loc[0]][loc[1]][0] = mobility[loc[0]][loc[1]][0] + 1
                    x_last = loc[0]
                    y_last = loc[1]
                else:
                    pass
    return mobility