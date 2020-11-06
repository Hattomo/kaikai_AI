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
import convolutional_neural_network as cnn
(train_data, train_label), (test_data, test_label) = mnist.load_data()

class MyCNN(cnn.Convolution_Neural_Network):

    def __init__(self, train_x, train_y):
        super().__init__(train_x, train_y)
        #CNNの構造はいろいろ変えたいから手動で書く
        self.conv = cl.Convolution_Layer(1, [3, 3])
        self.pool = pl.Pooling_Layer([2, 2])
        # self.fc = nn.Full_Neural_Network([3, 3, 1])

    def train(self):
        #CNNの構造はいろいろ変えたいから手動で書く
        self.conv(self.train_x[0])
        self.pool(self.conv.out)
        self.out = self.pool.out
        super().train()
# conv = cl.Convolution_Layer(1, [3, 3])
# pool = pl.Pooling_Layer([2, 2])

# conv(train_data[0])
# pool(conv.out)

# print(pool.out)


mycnn = MyCNN(train_data,train_label)
mycnn.train()