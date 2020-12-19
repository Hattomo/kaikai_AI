import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import setting

class Fully_Connect_Layer(nn.Neural_Network):

    def __init__(self, structure):
        super().__init__(structure)

    def reset(self,w_method="xavier"):
        self.weight = setting.set_weight(self.structure, w_method)
        self.cost, self.accurancy = list(), list()

    # train in dnn and get error
    def train(self, input_data, train_label):
        (batch, channel, height, width) = np.shape(input_data)
        up_error = np.zeros([batch, channel * height * width])
        train_data = np.zeros([batch, input_data.size // batch])
        # flatten training data
        for i in range(batch):
            train_data[i] = input_data[i].flatten()
        # forward propagation
        super().forwardpropagation(train_data, batch)
        # back propagation
        up_error = super().backpropagation(train_data, train_label)
        return up_error.reshape(batch, channel, height, width)

    def test(self, input_data, train_label, mode="abs"):
        (batch, channel, height, width) = np.shape(input_data)
        train_data = np.zeros([batch, channel * height * width])
        for i in range(batch):
            train_data[i] = input_data[i].flatten()
        super().test(train_data, train_label, mode)
