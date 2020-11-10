import sys

import numpy as np

class Pooling_Layer:

    def __init__(self, pooling_size, pooling_method="max-pooling"):
        self.poolfunc = self.__set_pooling_method(pooling_method)
        self.pooling_size = pooling_size
        self.index = []

    def __get_index(self, partical_data):
        max_ = np.argmax(partical_data)
        return (max_ // self.pooling_size[0], max_ % self.pooling_size[1])

    def __set_pooling_method(self, pooling_method):
        if pooling_method == "max-pooling":
            return np.max
        elif pooling_method == "mean-pooling":
            return np.average
        sys.stdout.write("Error: The pooling method is not found\n")
        sys.exit(1)

    def forwordpropagation(self, train_data):
        (data_channel, data_height, data_width) = np.shape(train_data)
        (pooling_height, pooling_width) = np.shape(self.pooling_size)
        # check pooling size
        if data_height % pooling_height != 0 or width % pooling_width != 0:
            sys.stdout.write("Error: The pooling_size is not right\n")
            sys.exit(1)
        # pooling
        p_result_height = data_height // pooling_height
        p_result_width = data_width // pooling_width
        p_result = np.zeros([data_channel, p_result_height, p_result_width])
        for h in range(data_channel):
            for i in range(p_result_height):
                for j in range(p_result_width):
                    partical_data = train_data[h][i * pooling_height:(i+1) * pooling_height,
                                                  j * pooling_width:(j+1) * pooling_width]
                    p_result[h][i][j] = self.poolfunc(partical_data)
                    self.index.append(self.__get_index(partical_data))
        return p_result

    def backpropagation(self, input_error):
        (input_channel, input_height, input_width) = np.shape(input_error)
        (pooling_height, pooling_width) = np.shape(self.pooling_size)
        output_error = np.zeros([input_channel, input_height * pooling_height, width * pooling_width])
        count = 0
        for h in range(input_channel):
            for i in range(input_height // pooling_height):
                for j in range(input_width // pooling_width):
                    output_error[h][i*pooling_height + self.index[count][0],
                                    j*pooling_width + self.index[count][1]] = input_error[h][i][j]
                    count += 1
        return output_error