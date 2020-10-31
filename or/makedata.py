import numpy as np
import random

# データの作成
'''
num データの数
data 0,1番目が学習データ 2番目が答え
'''
def dset(num):
    data = np.zeros((4 * num, 3))
    for i in range(num - 1):
        data[4 * i] = [0, 0, 0]
        data[4 * i + 1] = [0, 1, 1]
        data[4 * i + 2] = [1, 0, 1]
        data[4 * i + 3] = [1, 1, 1]
    return data
