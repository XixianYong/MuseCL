import numpy as np


def emb_fusion(a, b, c, method):
    if method == 'rv':
        return a
    if method == 'sv':
        return b
    if method == 'POI':
        return c
    if method == 'concat':
        return np.concatenate((np.array(a), np.array(b), np.array(c)), axis=1)
    if method == 'sv+rv':
        return np.concatenate((np.array(a), np.array(b)), axis=1)
    if method == 'fusion':
        return np.load("").reshape(517, 128)
    if method == 'svrv':
        return a+b