import sys
import random

import numpy as np

sys.path.append('./dataset')
import mnist

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
    elif d_name == "mnist_train" or d_name == "mnist_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data()
        data = np.zeros((num, 784 + 10), dtype=np.float128)
        for i in range(num):
            for j in range(28):
                for k in range(28):
                    if (d_name == "mnist_train"):
                        data[i][28*j + k] = train_data[i][j][k]
                    elif (d_name == "mnist_test"):
                        data[i][28*j + k] = test_data[i][j][k]
            for l in range(784, 794):
                if (d_name == "mnist_train"):
                    if (l - 784 == train_label[i]):
                        data[i][l] = 1.0
                    else:
                        data[i][l] = 0.0
                elif (d_name == "mnist_test"):
                    if (l - 784 == train_label[i]):
                        data[i][l] = 1.0
                    else:
                        data[i][l] = 0.0
        return data
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)
