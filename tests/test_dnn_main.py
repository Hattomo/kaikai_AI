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
    structure = [3, 3, 2]
    dropout = [0, 0.5, 0]
    batch = 4
    epoch = 10
    logic = "or"
    # set data
    (trainData, trainLabel) = lc.dset(logic, epoch * batch // 4)
    (testData, testLabel) = lc.dset(logic, 5)

    # ニューラルネットワークの生成
    orNN = nn.Neural_Network(structure, batch, dropout)
    # 学習
    for i in range(epoch):
        orNN.train(trainData, trainLabel)

def test_main_or_without_dropout():
    structure = [3, 3, 2]
    dropout = [0, 0, 0]
    batch = 4
    epoch = 10
    logic = "or"
    # set data
    (trainData, trainLabel) = lc.dset(logic, epoch * batch // 4)
    (testData, testLabel) = lc.dset(logic, 5)
    # ニューラルネットワークの生成
    orNN = nn.Neural_Network(structure, batch, dropout)
    # 学習
    for i in range(epoch):
        orNN.train(trainData, trainLabel)