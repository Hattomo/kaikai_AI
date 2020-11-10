import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import cnn_analysistool as catool
import convolution_layer as cl
import convolutional_neural_network as cnn
import csetting
import fully_connenct_layer as fc
import neural_network as nn
import pooling_layer as pl
import mnist

(train_data, train_label), (test_data, test_label) = mnist.load_data()

a = np.zeros([1, 4, 4])
count = 0
for i in range(4):
    for j in range(4):
        a[0][i][j] = count
        count += 1

error = np.array([[[-1,2],[3,4]]])

conv = cl.Convolution_Layer(1, 3, "test")

conv.forwordpropagation(a)
conv_out = conv.backpropagation(error)

print(conv_out)
