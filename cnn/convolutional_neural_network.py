import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import convolution_layer as cl
import convolutional_neural_network as cnn
import fully_connenct_layer as fc
import pooling_layer as pl

class Convolutional_Neural_Network:

    def __init__(self, structure):
        self.structure = structure

    def __call__(self, train_data, train_label):
        conv = cl.Convolution_Layer(1, 3)
        pool = pl.Pooling_Layer([2, 2])
        myfc = fc.Fully_Connect_Layer([16 + 1, 5, 1])

        conv_out = conv.convolution(train_data)
        pool_out = pool.pooling(conv_out)
        input_data = np.array(pool_out)
        error = myfc(input_data, train_label)
        pool_error = pool.backpooling(error)
        conv_error = conv.backconvolution(pool_error)
        print(conv_error)