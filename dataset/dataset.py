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
def logic(data_name, datasetsize, batchsize=1, data_error=0.0):
    # confirm batchsize
    if datasetsize % batchsize != 0:
        sys.stdout.write("Error : batch size is not good")
        sys.exit(10)
    # set data error
    a = -data_error * 100 / 2
    b = data_error * 100 / 2
    if data_name == "or":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    elif data_name == "and":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    elif data_name == "xor":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)
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

def img_data(data_name, datasetsize, batchsize=1, data_error=0.0):
    # confirm batchsize
    if datasetsize % batchsize != 0:
        sys.stdout.write("Error : batch size is not good")
        sys.exit(10)
    # set data error
    a = -data_error * 100 / 2
    b = data_error * 100 / 2
    if data_name == "cnn_ex":
        channel = 1
        data_height,data_width = 4,4
        labelsize = 4
        original_data = np.array([[[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0], [255, 0, 0, 255]],
                                    [[255, 255, 255, 255], [255, 0, 0, 255], [255, 0, 0, 255], [255, 255, 255, 255]],
                                    [[0, 255, 255, 0], [255, 255, 255, 255], [255, 255, 255, 255], [0, 255, 255, 0]],
                                    [[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]]])
        original_label = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    elif d_name == "cnn_exs":
        channel = 1
        data_height,data_width = 4,4
        labelsize = 2
        original_data = np.array([[[255, 255, 255, 255], [255, 255, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [255, 255, 255, 255], [255, 255, 255, 255]]])
        original_label = np.array([[0, 1],[1, 0]])
    # elif d_name == "dnn_ex":
    #     datasize = 16
    #     labelsize = 4
    #     original_data = np.array([[255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255], 
    #                               [255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255],
    #                               [0, 0, 255, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0],
    #                               [0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 255]])
    #     original_data = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)
    # make data and label
    data = np.zeros([datasetsize, data_height,data_width])
    label = np.zeros([datasetsize, labelsize])
    for i in range(datasetsize):
        data[i] = original_data[i % 4] + random.randint(a, b) / 100
        label[i] = original_label[i % 4]
    return data.reshape([datasetsize // batchsize, batchsize,channel,
                         data_height,data_width]), label.reshape([datasetsize // batchsize, batchsize, labelsize])

def data_shuffle(data, label):
    shuffle = np.random.permutation(len(label))
    data = data[shuffle]
    label = label[shuffle]

def logictest(data_name, testsize=1, data_error=0.0):
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
    elif data_name == "xor":
        datasize = labelsize = 2
        original_data = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        original_label = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
    # make data and label
    data = np.zeros([testsize, datasize])
    label = np.zeros([testsize, labelsize])
    for i in range(testsize):
        data[i] = original_data[i % 4] + random.randint(a, b) / 100
        label[i] = original_label[i % 4]
    return data, label
