import sys
import random

import numpy as np

# データの作成
# num データの数
# data 0,1番目が学習データ 2番目が答え
def dset(d_name, num):
    if d_name == "or":
        data = np.zeros((4 * num, 3))
        for i in range(num):
            data[4 * i] = [0, 0, 0]
            data[4*i + 1] = [0, 1, 1]
            data[4*i + 2] = [1, 0, 1]
            data[4*i + 3] = [1, 1, 1]
        return data
    elif d_name == "and":
        data = np.zeros((4 * num, 3))
        for i in range(num):
            data[4 * i] = [0, 0, 0]
            data[4*i + 1] = [0, 1, 0]
            data[4*i + 2] = [1, 0, 0]
            data[4*i + 3] = [1, 1, 1]
        return data
    elif d_name == "nand":
        data = np.zeros((4 * num, 3))
        for i in range(num):
            data[4 * i] = [0, 0, 1]
            data[4*i + 1] = [0, 1, 1]
            data[4*i + 2] = [1, 0, 1]
            data[4*i + 3] = [1, 1, 0]
        return data
    elif d_name == "xor":
        data = np.zeros((4 * num, 3))
        for i in range(num):
            data[4 * i] = [0, 0, 1]
            data[4*i + 1] = [0, 1, 0]
            data[4*i + 2] = [1, 0, 0]
            data[4*i + 3] = [1, 1, 1]
        return data
    elif d_name == "w_not":
        data = np.zeros((4 * num, 4))
        for i in range(num):
            data[4 * i] = [0, 0, 1, 1]
            data[4*i + 1] = [0, 1, 1, 0]
            data[4*i + 2] = [1, 0, 0, 1]
            data[4*i + 3] = [1, 1, 0, 0]
        return data
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)
