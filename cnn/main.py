import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import analysistool as atool
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

(trainData, trainLabel) = lc.dset("mnist8_direct_train", 50)
(testData, testLabel) = lc.dset("mnist8_direct_test", 1)

conv = cl.Convolution_Layer(in_channel=1, out_channel=8, ksize=3, pad=1)
conv2 = cl.Convolution_Layer(in_channel=8, out_channel=16, ksize=3, pad=1)
pool = pl.Pooling_Layer(pooling_size=[2, 2])
fullc = fc.Fully_Connect_Layer([256 + 1, 10, 10])
epoch = 300
for i in range(epoch):
    # train
    conv_out = conv.forwardpropagation(trainData / 255)
    conv_out2 = conv2.forwardpropagation(conv_out)
    pool_out = pool.forwardpropagation(conv_out2)
    normalized_data = normalize.normalize(pool_out)
    error = fullc.train(normalized_data, trainLabel)
    pool_error = pool.backpropagation(error)
    error = conv2.backpropagation(pool_error)
    conv.backpropagation(error)
    # test
    conv_out = conv.forwardpropagation(testData / 255)
    conv_out2 = conv2.forwardpropagation(conv_out)
    pool_out = pool.forwardpropagation(conv_out2)
    fullc.test(pool_out, trainLabel)

# draw graph
atool.draw(fullc.cost)
atool.accurancygraph(fullc.accurancy)