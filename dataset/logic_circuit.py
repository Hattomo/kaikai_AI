import sys
import random

import numpy as np

sys.path.append('./data')
import mnist

# データの作成
# num データの数
# data 0,1番目が学習データ 2番目が答え
def dset(d_name, num):
    if d_name == "or":
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        label = np.zeros((4 * num, 2))
        for i in range(num):
            label[4 * i] = [1, 0]
            label[4*i + 1] = [0, 1]
            label[4*i + 2] = [0, 1]
            label[4*i + 3] = [0, 1]
        return data, label
    elif d_name == "and":
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        label = np.zeros((4 * num, 2))
        for i in range(num):
            label[4 * i] = [1, 0]
            label[4*i + 1] = [1, 0]
            label[4*i + 2] = [1, 0]
            label[4*i + 3] = [0, 1]
        return data, label
    elif d_name == "nand":
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        label = np.zeros((4 * num, 2))
        for i in range(num):
            label[4 * i] = [0, 1]
            label[4*i + 1] = [0, 1]
            label[4*i + 2] = [0, 1]
            label[4*i + 3] = [1, 0]
        return data, label
    elif d_name == "xor":
        data = np.zeros((4 * num, 2))
        for i in range(num):
            data[4 * i] = [0, 0]
            data[4*i + 1] = [0, 1]
            data[4*i + 2] = [1, 0]
            data[4*i + 3] = [1, 1]
        label = np.zeros((4 * num, 2))
        for i in range(num):
            label[4 * i] = [1, 0]
            label[4*i + 1] = [0, 1]
            label[4*i + 2] = [0, 1]
            label[4*i + 3] = [1, 0]
        return data, label
    elif d_name == "mnist_train" or d_name == "mnist_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data()
        if d_name == "mnist_train":
            return train_data[:num], train_label[:num]
        elif d_name == "mnist_test":
            return test_data[:num], test_label[:num]
    elif d_name == "mnist16_mean_train" or d_name == "mnist16_mean_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data("mnist16_mean")
        if d_name == "mnist16_mean_train":
            return train_data[:num], train_label[:num]
        elif d_name == "mnist16_mean_test":
            return test_data[:num], test_label[:num]
    elif d_name == "mnist16_direct_train" or d_name == "mnist16_direct_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data("mnist16_direct")
        if d_name == "mnist16_direct_train":
            return train_data[:num], train_label[:num]
        elif d_name == "mnist16_direct_test":
            return test_data[:num], test_label[:num]
    elif d_name == "cnn_ex":
        data = np.zeros((4 * num, 1, 4, 4))
        for i in range(num):
            data[4 * i] = [[[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0], [255, 0, 0, 255]]]
            data[4*i + 1] = [[[255, 255, 255, 255], [255, 0, 0, 255], [255, 0, 0, 255], [255, 255, 255, 255]]]
            data[4*i + 2] = [[[0, 0, 255, 0], [255, 255, 255, 255], [0, 0, 255, 0], [0, 0, 255, 0]]]
            data[4*i + 3] = [[[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 255]]]
        label = np.zeros((4 * num, 4))
        for i in range(num):
            label[4 * i] = [1, 0, 0, 0]
            label[4*i + 1] = [0, 1, 0, 0]
            label[4*i + 2] = [0, 0, 1, 0]
            label[4*i + 3] = [0, 0, 0, 1]
        return data, label
    elif d_name == "cnn_exs":
        data = np.zeros((2 * num, 1, 4, 4))
        for i in range(num):
            data[2 * i] = [[[255, 255, 255, 255], [255, 255, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]]]
            data[2*i + 1] = [[[0, 0, 0, 0], [0, 0, 0, 0], [255, 255, 255, 255], [255, 255, 255, 255]]]
        label = np.zeros((2 * num, 2))
        for i in range(num):
            label[2 * i] = [0, 1]
            label[2*i + 1] = [1, 0]
        return data, label
    elif d_name == "dnn_ex":
        data = np.zeros((4 * num, 16))
        for i in range(num):
            data[4 * i] = [255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255]
            data[4*i + 1] = [255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255]
            data[4*i + 2] = [0, 0, 255, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0]
            data[4*i + 3] = [0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 255]
        label = np.zeros((4 * num, 4))
        for i in range(num):
            label[4 * i] = [1, 0, 0, 0]
            label[4*i + 1] = [0, 1, 0, 0]
            label[4*i + 2] = [0, 0, 1, 0]
            label[4*i + 3] = [0, 0, 0, 1]
        return data, label
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)

def data_shuffle(data, label):
    shuffle = np.random.permutation(len(label))
    data = data[shuffle]
    label = label[shuffle]
