import sys

import numpy as np

sys.path.append('./dnn')
import neural_network as nn

class Fully_Connect_Layer(nn.Neural_Network):

    def __init__(self, structure, batch):
        super().__init__(structure, batch)

    # train in dnn and get error
    def train(self, input_data, train_label):
        (batch, channel, height, width) = np.shape(input_data)
        up_error = np.zeros([batch, channel * height * width])
        train_data = np.zeros([batch, input_data.size // batch])
        for i in range(batch):
            train_data[i] = input_data[i].flatten()
        super().forwardpropagation(train_data)
        up_error = super().backpropagation(train_data, train_label)
        return up_error.reshape(batch, channel, height, width)

    def test(self, input_data, train_label):
        (batch, channel, height, width) = np.shape(input_data)
        train_data = np.zeros([batch, channel * height * width])
        for i in range(batch):
            train_data[i] = input_data[i].flatten()
        super().test(train_data, train_label)