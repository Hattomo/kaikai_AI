import sys

import numpy as np
import pytest

sys.path.append('./cnn')
sys.path.append("./shared")
import convolution_layer as cl

def test_concolution_layer_forwordpropagation():
    a = np.zeros([1, 4, 4])
    count = 0
    for i in range(4):
        for j in range(4):
            a[0][i][j] = count
            count += 1
    conv = cl.Convolution_Layer(1, 2, "test")
    out = conv.forwordpropagation(a)
    ans = np.array([[3.4, 4.4, 5.4], [7.4, 8.4, 9.4], [11.4, 12.4, 13.4]])
    assert ((out - ans) < 1e-3).all()

def test_convolution_layer_backpropagation():
    a = np.zeros([1, 4, 4])
    count = 0
    for i in range(4):
        for j in range(4):
            a[0][i][j] = count
            count += 1
    error = np.array([[[-1, 2], [3, 4]]])
    conv = cl.Convolution_Layer(1, 3, "test")
    # set train_data on convolution layer
    conv.forwordpropagation(a)
    out = conv.backpropagation(error)
    ans = np.array([[[1.08, 2.81, 4.34, 4.31], [5.01, 10.605, 13.98, 12.315], [12.99, 24.105, 27.48, 22.335],
                     [17.96, 30.89, 33.86, 25.83]]])
    assert ((out - ans) < 1e-3).all()
