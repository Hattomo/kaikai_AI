import sys
import random

import numpy as np

sys.path.append('./data')
import mnist

# データの作成
def logic(data_name, datasetsize, batchsize=1, data_error=0.0, testsize=10):
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
    data = np.zeros([datasetsize + testsize, datasize])
    label = np.zeros([datasetsize + testsize, labelsize])
    for i in range(datasetsize + testsize):
        data[i] = original_data[i % len(original_data)] + random.randint(a, b) / 100
        label[i] = original_label[i % len(original_label)]
    return data[:datasetsize].reshape([datasetsize // batchsize, batchsize, datasize
                                      ]), label[:datasetsize].reshape([datasetsize // batchsize, batchsize, labelsize
                                                                      ]), data[datasetsize:], label[datasetsize:]

def image_mnist(data_name, datasetsize, batchsize=1, testsize=10, isshuffle=False):
    if data_name == "mnist":
        # load data
        (train_data, train_label), (test_data, test_label) = mnist.load_data(data_name)
    elif data_name == "mnist16_mean":
        (train_data, train_label), (test_data, test_label) = mnist.load_data(data_name)
    elif data_name == "mnist16_direct":
        (train_data, train_label), (test_data, test_label) = mnist.load_data(data_name)
    elif data_name == "mnist8_mean":
        (train_data, train_label), (test_data, test_label) = mnist.load_data(data_name)
    elif data_name == "mnist8_direct":
        (train_data, train_label), (test_data, test_label) = mnist.load_data(data_name)
    channel = 1
    num, height, width = np.shape(train_data)
    # shuffle data
    if isshuffle:
        train_data, train_label = data_shuffle(train_data, train_label)
        test_data, test_label = data_shuffle(test_data, test_label)
    return train_data[:datasetsize].reshape([datasetsize // batchsize, batchsize, channel, height, width]), num2vec(
        train_label[:datasetsize]), test_data[:testsize], num2vec(test_label[:testsize])

def flatten_image(data_name, datasetsize, batchsize=1, data_error=0.0, testsize=10):
    # confirm batchsize
    if datasetsize % batchsize != 0:
        sys.stdout.write("Error : batch size is not good")
        sys.exit(10)
    # set data error
    a = -data_error * 100 / 2
    b = data_error * 100 / 2
    if data_name == "dnn_ex":
        datasize = 16
        labelsize = 4
        original_data = np.array([[255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255],
                                  [255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255],
                                  [0, 0, 255, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0],
                                  [0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 255]])
        original_label = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # make data and label
    data = np.zeros([datasetsize + testsize, datasize])
    label = np.zeros([datasetsize + testsize, labelsize])
    for i in range(datasetsize + testsize):
        data[i] = original_data[i % len(original_data)] + random.randint(a, b) / 100
        label[i] = original_label[i % len(original_label)]
    return data[:datasetsize].reshape([datasetsize // batchsize, batchsize, datasize
                                      ]), label[:datasetsize].reshape([datasetsize // batchsize, batchsize, labelsize
                                                                      ]), data[datasetsize:], label[datasetsize:]

def image(data_name, datasetsize, batchsize=1, data_error=0.0, testsize=10):
    # confirm batchsize
    if datasetsize % batchsize != 0:
        sys.stdout.write("Error : batch size is not good")
        sys.exit(10)
    # set data error
    a = -data_error * 100 / 2
    b = data_error * 100 / 2
    if data_name == "cnn_ex":
        channel = 1
        data_height, data_width = 4, 4
        labelsize = 4
        original_data = np.array([[[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0], [255, 0, 0, 255]],
                                  [[255, 255, 255, 255], [255, 0, 0, 255], [255, 0, 0, 255], [255, 255, 255, 255]],
                                  [[0, 255, 255, 0], [255, 255, 255, 255], [255, 255, 255, 255], [0, 255, 255, 0]],
                                  [[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]]])
        original_label = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif data_name == "cnn_exs":
        channel = 1
        data_height, data_width = 4, 4
        labelsize = 2
        original_data = np.array([[[255, 255, 255, 255], [255, 255, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                                  [[0, 0, 0, 0], [0, 0, 0, 0], [255, 255, 255, 255], [255, 255, 255, 255]]])
        original_label = np.array([[0, 1], [1, 0]])
    else:
        sys.stdout.write("Error: the data name is not found\n")
        sys.exit(1)
    # make data and label
    data = np.zeros([datasetsize + testsize, channel, data_height, data_width])
    label = np.zeros([datasetsize + testsize, labelsize])
    for i in range(datasetsize + testsize):
        data[i] = original_data[i % len(original_data)] + random.randint(a, b) / 100
        label[i] = original_label[i % len(original_label)]
    data[:datasetsize].reshape([datasetsize, channel, data_height, data_width])
    return data[:datasetsize].reshape([datasetsize // batchsize, batchsize, channel, data_height, data_width
                                      ]), label[:datasetsize].reshape([datasetsize // batchsize, batchsize, labelsize
                                                                      ]), data[datasetsize:], label[datasetsize:]

def data_shuffle(data, label):
    shuffle = np.random.permutation(len(label))
    data = data[shuffle]
    label = label[shuffle]
    return data, label

def num2vec(numberlabel):
    label_max = 0
    for i in range(len(numberlabel)):
        if label_max < numberlabel[i]:
            label_max = numberlabel[i]
    veclabel = np.zeros([len(numberlabel), label_max])
    for i in range(len(numberlabel)):
        veclabel[i][numberlabel[i] - 1] = 1
    return veclabel