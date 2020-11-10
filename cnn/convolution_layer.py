import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import activationfunction as af
import csetting
"""
convolution
    mask (numpy) :  data_channel * N * M
    kernel (list[int]) : L * L
    stride (int) : s
    c_result (numpy) : channel * {( N - L + 1 )/s} * {( M - L + 1 )/s}

"""

class Convolution_Layer:

    def __init__(self,
                 channel,
                 kernel_size,
                 k_method="xivier",
                 stride=1,
                 actfunc="relu",
                 padding_method="valid-padding",
                 train_ratio=0.1):
        self.channel = channel
        self.stride = stride
        self.actfunc = af.relu
        self.diffact = af.diffrelu
        self.padding_method = padding_method
        self.kernel = self.__select_w(k_method, kernel_size)
        self.train_ratio = 0.1

    def __select_w(self, k_method, kernel_size):
        if k_method == "xivier":
            return csetting.knet(self.channel, kernel_size, csetting.xivier)
        elif k_method == "he":
            return csetting.knet(self.channel, kernel_size, csetting.he)
        sys.stdout.write("Error: The kernel method is not found\n")
        sys.exit(1)

    def __padding(self, padding_method, train_data):
        if padding_method == "valid-padding":
            return train_data
        sys.stdout.write("Error: The padding method is not found\n")
        sys.exit(1)

    def forwordpropagation(self, train_data):
        self.train_data = train_data
        padding_data = self.__padding(self.padding_method, train_data)
        return self.__forwordconvolution(padding_data)

    def __forwordconvolution(self, train_data):
        (data_channel, data_height, data_width) = np.shape(train_data)
        (kernel_channel, kernel_height, kernel_width) = np.shape(self.kernel)
        # check
        if ((data_height-kernel_height) % self.stride) or ((data_width-kernel_width) % self.stride):
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        c_result = self.__convolution(train_data, self.kernel)
        return self.actfunc(c_result)

    def backpropagation(self, input_error):
        error = self.__backconvolution(input_error)
        self.kernel = self.train_ratio * error
        return 

    def __backconvolution(self, input_error):
        z = self.diffact(input_error)
        return self.__convolution(self.train_data, z)

    def __convolution(self, mask, _filter):
        c_result_channel = len(mask)
        c_result_height = int((len(mask[0]) - len(_filter[0])) / self.stride) + 1
        c_result_width = int((len(mask[0][0]) - len(_filter[0][0])) / self.stride) + 1
        c_result = np.zeros([c_result_channel, c_result_height, c_result_width])
        for g in range(c_result_channel):
            for h in range(len(_filter)):
                for i in range(c_result_height):
                    for j in range(c_result_width):
                        y = mask[g][i:i + len(_filter[0]), j:j + len(_filter[0][0])] @ _filter[h]
                        c_result[h][i][j] = np.sum(y)
        return c_result
