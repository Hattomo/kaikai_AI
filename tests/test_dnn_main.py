import sys

import numpy as np
import pytest

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import numpy_files as npfiles
import logic_circuit as lc

def test_main_or_with_dropout():
    epoch = 10
    batch = 4
    td_num = 2
    tl_num = 2
    # set data
    trainData = np.ones([epoch, batch, td_num])
    trainLabel = np.zeros([epoch, batch, tl_num])
    # ニューラルネットワークの生成
    structure = [td_num + 1, 3, tl_num]
    dropout = [0, 0.5, 0]
    orNN = nn.Neural_Network(structure, batch, dropout)
    # 学習
    for i in range(epoch):
        orNN.train(trainData[i], trainLabel[i])

def test_main_or_without_dropout():
    epoch = 10
    batch = 4
    td_num = 2
    tl_num = 2
    # set data
    trainData = np.ones([epoch, batch, td_num])
    trainLabel = np.zeros([epoch, batch, tl_num])
    # ニューラルネットワークの生成
    structure = [td_num + 1, 3, tl_num]
    dropout = [0, 0, 0]
    orNN = nn.Neural_Network(structure, batch, dropout)
    # 学習
    for i in range(epoch):
        orNN.train(trainData[i], trainLabel[i])