import sys

import numpy as np

sys.path.append('./dnn')
import neural_network as nn

class Fully_Connect_Layer(nn.Neural_Network):

    def __init__(self, structure):
        super().__init__(structure)

    def __call__(self, input_data, train_label):
        data_size = np.shape(input_data)
        train_data = self.__flattened(input_data, data_size)
        return self.__train(train_data, train_label, data_size)

    #　平坦化
    def __flattened(self, input_data):
        return input_data.flatten()

    # train in dnn and get error
    def __train(self, train_data, train_label, output_size):
        (channel, height, width) = output_size
        # 学習
        super().forwardpropagation(train_data)
        # 誤差の伝播
        error = super().backpropagation(train_data, train_label, isexternal=True)
        return error.reshape([channel, height, width])

    def test(self, input_data, train_label):
        data_size = np.shape(input_data)
        train_data = self.__flattened(input_data)
        super().test(train_data, train_label)