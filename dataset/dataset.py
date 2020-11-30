import sys
import random

import numpy as np

sys.path.append('./data')
import mnist

def step(x):
    return 1.0 * (x > 0.0)

# データの作成
# num データの数
# data 0,1番目が学習データ 2番目が答え
def logic(d_name, datasetsize, batchsize=1, data_error=0.0):
    # confirm batchsize
    if datasetsize % batchsize != 0:
        sys.stdout.write("Error : batch size is not good")
        sys.exit(10)
    # set data error
    a = -data_error * 100 / 2
    b = data_error * 100 / 2
    if d_name == "or":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    elif d_name == "and":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    elif d_name == "xor":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
    # make data and label
    data = np.zeros([datasetsize, datasize])
    label = np.zeros([datasetsize, labelsize])
    for i in range(datasetsize):
        data[i] = original_data[i % 4] + random.randint(a, b) / 100
        label[i] = original_label[i % 4]
    return data.reshape([datasetsize // batchsize, batchsize,
                         datasize]), label.reshape([datasetsize // batchsize, batchsize, labelsize])

def mnist():
    if d_name == "mnist28_train" or d_name == "mnist28_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data("mnist28")
        if d_name == "mnist28_train":
            return train_data[:num], train_label[:num]
        elif d_name == "mnist28_test":
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
    elif d_name == "mnist8_mean_train" or d_name == "mnist8_mean_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data("mnist8_mean")
        if d_name == "mnist8_mean_train":
            return train_data[:num], train_label[:num]
        elif d_name == "mnist8_mean_test":
            return test_data[:num], test_label[:num]
    elif d_name == "mnist8_direct_train" or d_name == "mnist8_direct_test":
        (train_data, train_label), (test_data, test_label) = mnist.load_data("mnist8_direct")
        if d_name == "mnist8_direct_train":
            return train_data[:num], train_label[:num]
        elif d_name == "mnist8_direct_test":
            return test_data[:num], test_label[:num]

def img_data():
    if d_name == "cnn_ex":
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

def logictest(data_name, testsize=1,data_error=0.0):
    a = -data_error * 100 / 2
    b = data_error * 100 / 2
    if data_name == "or":
        datasize = labelsize = 2
        original_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        original_label = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
    elif data_name == "and":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    # make data and label
    data = np.zeros([testsize, datasize])
    label = np.zeros([testsize, labelsize])
    for i in range(testsize):
        data[i] = original_data[i % 4] + random.randint(a, b) / 100
        label[i] = original_label[i % 4]
    return data, label
