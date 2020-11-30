import sys

import numpy as np
import pytest

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")
import neural_network as nn

def test_forwordpropagation():
    pass

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

def test_compare():
    pass