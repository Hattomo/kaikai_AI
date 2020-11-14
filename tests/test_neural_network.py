import sys

import numpy as np
import pytest

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import activationfunction as af
import neural_network as nn
import logic_circuit as lc

def test_forwordpropagation():
    pass

def test_backpropagation():
    pass

def test_dropout_shake():
    structure = [3, 4, 2]
    dropout = [0, 0.9, 0]
    epoch = 1
    logic = "or"
    # set data
    trainData, trainLabel = lc.dset(logic, epoch)
    testData, testLabel = lc.dset(logic, 10)
    # ニューラルネットワークの生成
    dnn = nn.Neural_Network(structure, dropout, "he", "sigmoid")
    dnn.do[0] = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    dnn._Neural_Network__dropout_shake(False)
    assert np.all(dnn.do[0] == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    assert np.all(dnn.do[1] == np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]))

def test_compare():
    pass