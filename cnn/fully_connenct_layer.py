import sys

import numpy as np

sys.path.append('./dnn')
import neural_network as nn

class Fully_Connect_Layer(nn.Neural_Network):

    def __init__(self, structure):
        super().__init__(structure)

    def __call__(self, input_data, train_label):
        train_data = self.flattend(input_data)
        return self.train(input_data, train_label)

    #　平坦化
    def flattened(self,input_data):
        channel = len(input_data)
        height = len(input_data[0])
        width = len(input_data[0][0])
        train_data = np.zeros(channel * height * width)
        count = 0
        for h in range(channel):
            for i in range(height):
                for j in range(width):
                    train_data[count] = input_data[h][i][j]
                    count += 1
        return train_data
        
    # train in dnn and get error
    def train(self, train_data, train_label):
        # 学習
        super().forwordpropagation(train_data)
        # 誤差の伝播
        error = super().backpropagation(train_data, train_label, flag=True)
        return error.reshape([channel, height, width])