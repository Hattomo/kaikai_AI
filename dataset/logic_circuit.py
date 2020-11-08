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
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1))
        for i in range(4 * num):
            label[i] = data[i][0] or data[i][1]
        dataset.append(label)
        return dataset
    elif d_name == "and":
        dataset = list()
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1))
        for i in range(4 * num):
            label[i] = data[i][0] and data[i][1]
        dataset.append(label)
        return dataset
    elif d_name == "nand":
        dataset = list()
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1))
        for i in range(4 * num):
            label[i] = not (data[i][0] and data[i][1])
        dataset.append(label)
        return dataset
    elif d_name == "xor":
        dataset = list()
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 1))
        for i in range(4 * num):
            label[i] = int(data[i][0]) ^ int(data[i][1])
        dataset.append(label)
        return dataset
    elif d_name == "w_not":
        dataset = list()
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        dataset.append(data)
        label = np.zeros((4 * num, 2))
        for i in range(num):
            label[4 * i] = [1, 1]
            label[4*i + 1] = [1, 0]
            label[4*i + 2] = [0, 1]
            label[4*i + 3] = [0, 0]
        dataset.append(label)
        return dataset
    elif d_name == "mnist_train" or d_name == "mnist_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data()
        if d_name == "mnist_train":
            return (train_data[:num], train_label[:num])
        elif d_name == "mnist_test":
            return (test_data[:num], test_label[:num])
    elif d_name == "cnn_ex":
        dataset = []
        data = np.zeros((4 * num, 16))
        for i in range(num):
            data[4 * i] = [[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0], [255, 0, 0, 255]]
            data[4*i + 1] = [[255, 255, 255, 255], [255, 0, 0, 255], [255, 0, 0, 255], [255, 255, 255, 255]]
            data[4*i + 2] = [[0, 0, 255, 0], [255, 255, 255, 255], [0, 0, 255, 0], [0, 0, 255, 0]]
            data[4*i + 3] = [[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 255]]
        dataset.append(data)
        label = np.zeros((4 * num, 4))
        for i in range(num):
            label[4 * i] = [1, 0, 0, 0]
            label[4*i + 1] = [0, 1, 0, 0]
            label[4*i + 2] = [0, 0, 1, 0]
            label[4*i + 3] = [0, 0, 0, 1]
        dataset.append(data)
        return dataset
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)
