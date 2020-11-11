import sys

import numpy as np
import pytest

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import activationfunction as af
import neural_network as nn
import logic_circuit as lc

def test_main_or():
    structure = [3, 3, 2]
    epoch = 5
    logic = "or"
    # set data
    (trainData, trainLabel) = lc.dset(logic, epoch)
    (testData, testLabel) = lc.dset(logic, 5)
    # ニューラルネットワークの生成
    orNN = nn.Neural_Network(structure, "he", "tanh")
    # 学習
    count = 5
    for i in range(count):
        orNN.train(trainData, trainLabel)
        orNN.test(testData, testLabel)