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

batch = 8
epoch = 10

Data, Label = lc.dset("cnn_ex", epoch * batch // 4)

data_num, D_channel, D_height, D_width = np.shape(Data)
data_num, L_num = np.shape(Label)
trainData = Data.reshape([epoch, batch, D_channel, D_height, D_width])
trainLabel = Label.reshape([epoch, batch, -1])

conv = cl.Convolution_Layer(in_channel=1, out_channel=8, ksize=3, pad=1)
pool = pl.Pooling_Layer(pooling_size=[2, 2])
fullc = fc.Fully_Connect_Layer([32 + 1, 10, 4], batch)
normalize = nl.Normalization_Layer()

for i in range(epoch):
    conv_out = conv.forwardpropagation(trainData[i])
    pool_out = pool.forwardpropagation(conv_out)
    normalized_data = normalize.normalize(pool_out)
    error = fullc.train(normalized_data, trainLabel[i])
    pool_error = pool.backpropagation(error)
    conv.backpropagation(pool_error)
    fullc.test(normalized_data, trainLabel[i])
