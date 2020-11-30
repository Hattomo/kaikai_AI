import sys

import numpy as np

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles

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
