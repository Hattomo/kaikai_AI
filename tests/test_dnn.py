import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import neural_network as nn

def test_main_or_with_dropout():
    datasetsize = 8
    batchsize = 4
    td_num = 2
    tl_num = 2
    # set data
    trainData = np.ones([datasetsize, batchsize, td_num])
    trainLabel = np.zeros([datasetsize, batchsize, tl_num])
    # ニューラルネットワークの生成
    structure = [td_num + 1, 3, tl_num]
    dropout = [0, 0.5, 0]
    myNN = nn.Neural_Network(structure, dropout)
    # 学習
    epoch = 10
    for i in range(epoch):
        myNN.train(trainData, trainLabel)

def test_main_or_without_dropout():
    datasetsize = 8
    batchsize = 4
    td_num = 2
    tl_num = 2
    # set data
    trainData = np.ones([datasetsize, batchsize, td_num])
    trainLabel = np.zeros([datasetsize, batchsize, tl_num])
    # ニューラルネットワークの生成
    structure = [td_num + 1, 3, tl_num]
    myNN = nn.Neural_Network(structure)
    epoch = 10
    # 学習
    for i in range(epoch):
        myNN.train(trainData, trainLabel)

def test_forwordpropagation():
    batchsize = 4
    td_num = 2
    tl_num = 2
    test = np.array([[0.51301977, 0.52648607], [0.51301977, 0.52648607], [0.51301977, 0.52648607],
                     [0.51301977, 0.52648607]])
    # set data
    trainData = np.ones([batchsize, td_num])
    # ニューラルネットワークの生成
    structure = [td_num + 1, 3, tl_num]
    myNN = nn.Neural_Network(structure, w_method="test")
    # test
    myNN.forwardpropagation(trainData, batchsize)
    assert np.allclose(myNN.z[-1], test), "dnn/neural_network/forwardpropagaiton is output error"

def test_backpropagation():
    pass

def test_dropout_shake():
    structure = [3, 4, 2]
    dropout = [0, 0.9, 0]
    dnn = nn.Neural_Network(structure, dropout)
    dnn.do[0] = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    dnn._Neural_Network__dropout_shake(False)
    assert np.all(dnn.do[0] == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    assert np.all(dnn.do[1] == np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]))