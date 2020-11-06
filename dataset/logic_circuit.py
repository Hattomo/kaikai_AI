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
        dataset = list()
        data = np.zeros((4 * num, 2), dtype=np.float128)
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1), dtype=np.float128)
        for i in range(4 * num):
            label[i] = data[i][0] or data[i][1]
        dataset.append(label)
        return dataset
    elif d_name == "and":
        dataset = list()
        data = np.zeros((4 * num, 2), dtype=np.float128)
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1), dtype=np.float128)
        for i in range(4 * num):
            label[i] = data[i][0] and data[i][1]
        dataset.append(label)
        return dataset
    elif d_name == "nand":
        dataset = list()
        data = np.zeros((4 * num, 2), dtype=np.float128)
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1), dtype=np.float128)
        for i in range(4 * num):
            label[i] = !(data[i][0] and data[i][1])
        dataset.append(label)
        return dataset
    elif d_name == "xor":
        dataset = list()
        data = np.zeros((4 * num, 2), dtype=np.float128)
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1), dtype=np.float128)
        for i in range(4 * num):
            label[i] = data[i][0] ^ data[i][1]
        dataset.append(label)
        return dataset
    elif d_name == "w_not":
        dataset = list()
        data = np.zeros((4 * num, 3), dtype=np.float128)
        for i in range(num):
            data[4 * i] = [0, 0, 1]
            data[4*i + 1] = [0, 1, 1]
            data[4*i + 2] = [1, 0, 0]
            data[4*i + 3] = [1, 1, 0]
        dataset.append(data)
        label = np.zeros((4 * num, 1), dtype=np.float128)
        for i in range(num)
            label[4 * i] = [1]
            label[4*i + 1] = [1]
            label[4*i + 2] = [0]
            label[4*i + 3] = [0]
        dataset.appen(data)
        return dataset
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
