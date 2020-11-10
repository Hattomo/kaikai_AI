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

conv = cl.Convolution_Layer(3, 3)
myfc = fc.Fully_Connect_Layer([4 + 1, 4, 4])

for i in range(1000):
    out = conv.forwordpropagation(trainData[i])
    error = myfc(out, trainLabel[i])
    conv.backpropagation(error)
    myfc.test(out, trainLabel)
