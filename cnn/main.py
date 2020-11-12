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
import logic_circuit as lc

(trainData, trainLabel) = lc.dset("cnn_ex", 1000)
(testData, testLabel) = lc.dset("cnn_ex", 1)

in_channel = 3
data_size = 4
a = np.zeros(in_channel*data_size**2)
for i in range(in_channel*data_size**2):
    a[i] = i
data = a.reshape([in_channel, data_size, data_size])


conv = cl.Convolution_Layer(in_channel=1,out_channel=3,ksize=3,pad=1,k_method="test")
pool = pl.Pooling_Layer(pooling_size=2)
myfc = fc.Fully_Connect_Layer([1,1,1])

conv_out = conv.forwardpropagation(data)

print(conv_out)
