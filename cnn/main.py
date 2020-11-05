import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import cnn_analysistool as catool
import convolution_layer as cl
import csetting
import neural_network as nn
import pooling_layer as pl
import mnist

(train_data, train_label), (test_data, test_label) = mnist.load_data()

a = np.zeros([10, 10])
for i in range(10):
    for j in range(10):
        a[i][j] = i

conv = cl.Convolution_Layer(a, 3, 3)
conv_out = conv.convolution()
pool = pl.Pooling_Layer(conv_out, [2, 2])
pool_out = pool.pooling()
print(pool_out)