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
import normalization_layer as nl
import pooling_layer as pl
import mnist
import logic_circuit as lc

(trainData, trainLabel) = lc.dset("cnn_ex", 5)
(testData, testLabel) = lc.dset("cnn_ex", 1)

conv = cl.Convolution_Layer(in_channel=1, out_channel=8, ksize=3, pad=1)
pool = pl.Pooling_Layer(pooling_size=[2, 2])
fullc = fc.Fully_Connect_Layer([32 + 1, 10, 4])
normalize = nl.Normalization_Layer()

epoch = 10
for i in range(epoch):
    conv_out = conv.forwardpropagation(trainData)
    pool_out = pool.forwardpropagation(conv_out)
    normalized_data = normalize.normalize(pool_out)
    error = fullc.train(normalized_data, trainLabel)
    pool_error = pool.backpropagation(error)
    conv.backpropagation(pool_error)
    fullc.test(pool_out, trainLabel)