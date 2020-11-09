import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import activationfunction as af
import csetting
"""
convolution
    train_data (numpy) :  data_channel*N*M
    kernel (list[int]) : L*L
    stride (int) : s
    c_result (numpy) : channel*{(N-L+1)/s}*{(N-L+1)/s}

padding(valid-padding)
    p_result (numpy) : N*N

backconvolution
    train_data (numpy) : N*N
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

    def forwordpropagation(self, train_data):
        self.train_data = train_data
        padding_data = self.__padding(self.padding_method, train_data)
        return self.__convolution(padding_data)

    def __padding(self, padding_method, train_data):
        #padding
        if padding_method == "valid-padding":
            return train_data
        sys.stdout.write("Error: The padding method is not found\n")
        sys.exit(1)

    # convolution
    def __convolution(self, train_data):
        data_height = len(train_data[0])
        data_width = len(train_data[0][0])
        kernel_size = len(self.kernel[0])
        # check
        if ((data_height-kernel_size) % self.stride) or ((data_width-kernel_size) % self.stride):
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        # make c_result
        # c_result_height = int((data_height - kernel_size) / self.stride + 1)
        # c_result_width = int((data_width - kernel_size) / self.stride + 1)
        # c_result = np.zeros([self.channel, c_result_height, c_result_width])
        # forword convolution
        c_result = self.convolution(train_data, self.kernel)
        c_result = self.actfunc(c_result)
        # for h in range(self.channel):
        #     for i in range(c_result_height):
        #         for j in range(c_result_width):
        #             y = train_data[0][i:i + kernel_size, j:j + kernel_size] @ self.kernel[h]
        #             z = np.sum(y)
        #             c_result[h][i][j] = self.actfunc(z)
        return c_result

    def backpropagation(self, input_error):
        error = self.__backconvolution(input_error)
        self.kernel = self.train_ratio * error
        return self.kernel

    # backconvolution
    def __backconvolution(self, input_error):
        input_size = len(input_error[0])
        data_channel = len(self.train_data)
        data_height = len(self.train_data[0])
        data_width = len(self.train_data[0][0])
        kernel_size = len(self.kernel)
        # make output_error array
        c_result_height = data_height
        c_result_width = data_width
        output_error = np.zeros([data_channel, kernel_size, kernel_size])
        # back convolution
        z = self.diffact(input_error)
        self.convolution(self.train_data, z)
        # for h in range(data_channel):
        #     for i in range(c_result_height - input_size + 1):
        #         for j in range(c_result_width - input_size + 1):
        #             z = self.diffact(input_error)
        #             output_error[h][i][j] = np.sum(self.train_data[h][i:i + input_size, j:j + input_size] @ z[h])
        return output_error

    def convolution(self, mask, _filter):
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
