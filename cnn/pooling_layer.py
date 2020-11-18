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
        (data_channel, data_height, data_width) = np.shape(train_data)
        # check pooling size
        if data_height % self.p_size[0] != 0 or data_width % self.p_size[1] != 0:
            sys.stdout.write("Error: The pooling_size is not right\n")
            sys.exit(1)
        # pooling
        p_result_height = data_height // self.p_size[0]
        p_result_width = data_width // self.p_size[1]
        p_result = np.zeros([data_channel, p_result_height, p_result_width])
        for h in range(data_channel):
            for i in range(p_result_height):
                for j in range(p_result_width):
                    partical_data = train_data[h][i * self.p_size[0]:(i+1) * self.p_size[0],
                                                  j * self.p_size[1]:(j+1) * self.p_size[1]]
                    p_result[h][i][j] = self.poolfunc(partical_data)
                    self.index.append(self.__get_index(partical_data))
        return p_result

    def backpropagation(self, input_error):
        (input_channel, input_height, input_width) = np.shape(input_error)
        output_error = np.zeros([input_channel, input_height * self.p_size[0], input_width * self.p_size[1]])
        count = 0
        for h in range(input_channel):
            for i in range(input_height):
                for j in range(input_width):
                    output_error[h][i * self.p_size[0] + self.index[count][0],
                                    j * self.p_size[1] + self.index[count][1]] = input_error[h][i][j]
                    count += 1
        return output_error