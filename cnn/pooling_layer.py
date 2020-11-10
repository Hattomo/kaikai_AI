import sys

import numpy as np

class Pooling_Layer:

    def __init__(self, pooling_size, pooling_method="max-pooling"):
        self.poolfunc = self.__set_pooling_method(pooling_method)
        self.pooling_size = pooling_size
        self.index = []

    # select max index
    def __get_index(self, partical_data):
        max_ = np.argmax(partical_data)
        return (int(max_ / self.pooling_size[0]), int(max_ % self.pooling_size[1]))

    def forwardpropagation(self, train_data):
        (channel, height, width) = np.shape(train_data)
        # check pooling size
        if width % self.pooling_size[0] != 0 or height % self.pooling_size[1] != 0:
            sys.stdout.write("Error: The pooling_size is not right\n")
            sys.exit(1)
        # pooling
        out_height = int(height / self.pooling_size[0])
        out_width = int(width / self.pooling_size[1])
        out = np.zeros([channel, out_height, out_width])
        for h in range(channel):
            for i in range(out_height):
                for j in range(out_width):
                    partical_data = train_data[h][i * self.pooling_size[0]:(i+1) * self.pooling_size[0],
                                                  i * self.pooling_size[1]:(i+1) * self.pooling_size[1]]
                    out[h][i][j] = self.poolfunc(partical_data)
                    self.index.append(self.__get_index(partical_data))
        return out

    def __set_pooling_method(self, pooling_method):
        if pooling_method == "max-pooling":
            return np.max
        elif pooling_method == "mean-pooling":
            return np.average
        sys.stdout.write("Error: The pooling method is not found\n")
        sys.exit(1)

    def backpropagation(self, input_error):
        channel = len(input_error)
        height = len(input_error[0]) * self.pooling_size[0]
        width = len(input_error[0][0]) * self.pooling_size[1]
        output_error = np.zeros([channel, height, width])
        count = 0
        for h in range(channel):
            for i in range(int(height / self.pooling_size[0])):
                for j in range(int(width / self.pooling_size[1])):
                    output_error[h][i * self.pooling_size[0] + self.index[count][0],
                                    j * self.pooling_size[1] + self.index[count][1]] = input_error[h][i][j]
                    count += 1
        return output_error