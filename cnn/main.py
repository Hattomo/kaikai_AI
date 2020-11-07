import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import cnn_analysistool as catool
import convolution_layer as cl
import csetting
import fully_connenct_layer as fc
import neural_network as nn
import pooling_layer as pl
import mnist

(train_data, train_label), (test_data, test_label) = mnist.load_data()

a = np.zeros([10, 10])
for i in range(10):
    for j in range(10):
        a[i][j] = i

conv = cl.Convolution_Layer(1, 3)
pool = pl.Pooling_Layer([2, 2])
myfc = fc.Fully_Connect_Layer([16 + 1, 5, 1])

conv_out = conv.convolution(a)
pool_out = pool.pooling(conv_out)
input_data = np.array(pool_out)

error = myfc(input_data, [1])
pool_error = pool.backpooling(error)
print(pool_error)