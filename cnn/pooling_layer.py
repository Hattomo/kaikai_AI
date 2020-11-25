import sys

import numpy as np

class Pooling_Layer:

    def __init__(self, pooling_size, pooling_method="max-pooling"):
        self.poolfunc = self.__set_pooling_method(pooling_method)
        self.p_size = pooling_size
        self.index = []

    def __get_index(self, partical_data):
        max_ = np.argmax(partical_data)
        return (max_ // self.p_size[0], max_ % self.p_size[1])

    def __set_pooling_method(self, pooling_method):
        if pooling_method == "max-pooling":
            return np.max
        elif pooling_method == "mean-pooling":
            return np.average
        sys.stdout.write("Error: The pooling method is not found\n")
        sys.exit(1)

    def forwardpropagation(self, train_data):
        self.index = list()
        (batch, data_channel, data_height, data_width) = np.shape(train_data)
        # check pooling size
        if data_height % self.p_size[0] != 0 or data_width % self.p_size[1] != 0:
            sys.stdout.write("Error: The pooling_size is not right\n")
            sys.exit(1)
        # pooling
        result_height = data_height // self.p_size[0]
        result_width = data_width // self.p_size[1]
        result = np.zeros([batch, data_channel, result_height, result_width])
        for h in range(batch):
            for i in range(data_channel):
                for j in range(result_height):
                    for k in range(result_width):
                        patch = train_data[h][i][j * self.p_size[0]:(j+1) * self.p_size[0],
                                                 k * self.p_size[1]:(k+1) * self.p_size[1]]
                        result[h][i][j][k] = self.poolfunc(patch)
                        self.index.append(self.__get_index(patch))
        # Batch Normalization
        result = (result - result.mean()) / (result.std() + 1e-9)
        result = result * (2 / (result.max() - result.min()))
        result -= (result.max() - 1)
        return result

    def backpropagation(self, input_error):
        (batch, input_channel, input_height, input_width) = np.shape(input_error)
        output_error = np.zeros([batch, input_channel, input_height * self.p_size[0], input_width * self.p_size[1]])
        count = 0
        for h in range(batch):
            for i in range(input_channel):
                for j in range(input_height):
                    for k in range(input_width):
                        output_error[h][i][j * self.p_size[0] + self.index[count][0],
                                           k * self.p_size[1] + self.index[count][1]] = input_error[h][i][j][k]
                        count += 1
        return output_error