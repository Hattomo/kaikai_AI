import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import activationfunction as af
import csetting
"""
convolution
    train_data (numpy) :  gray-scale, N*N
    kernel (list[int]) : L*L
    stride (int) : s
    c_result (numpy) : channel*{(N-L+1)/s}*{(N-L+1)/s}

padding(valid-padding)
    p_result (numpy) : N*N
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.actfunc = af.relu
        self.diffact = af.diffrelu
        self.padding_method = padding_method
        self.__select_w(k_method)
        self.train_ratio = 0.1

    # set kernel
    def __select_w(self, k_method):
        if k_method == "xivier":
            self.kernel = csetting.knet(self.channel, self.kernel_size, csetting.xivier)
        elif k_method == "he":
            self.kernel = csetting.knet(self.channel, self.kernel_size, csetting.he)
        else:
            sys.stdout.write("Error: The kernel method is not found\n")
            sys.exit(1)

    # forwordpropagation
    def convolution(self, train_data):
        self.train_data = train_data
        padding_data = self.__padding(train_data)
        return self.__convolution(padding_data)

    # padding
    def __padding(self, train_data):
        #padding
        if self.padding_method == "valid-padding":
            p_result = train_data
        else:
            sys.stdout.write("Error: The padding method is not found\n")
            sys.exit(1)
        return p_result

    # convolution
    def __convolution(self, train_data):
        data_size = len(train_data[0])
        # make output array
        if (data_size - self.kernel_size) % self.stride == 0:
            c_result_size = int((data_size - self.kernel_size) / self.stride + 1)
            c_result = np.zeros([self.channel, c_result_size, c_result_size])
        else:
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        #convolution
        for h in range(self.channel):
            for i in range(c_result_size):
                for j in range(c_result_size):
                    y = (train_data[i:i + self.kernel_size, j:j + self.kernel_size].T) @ self.kernel[h]
                    z = np.sum(y)
                    c_result[h][i][j] = self.actfunc(z) 
        return c_result

    # backpropagation
    def backconvolution(self, input_error):
        error = self.__backconvolution(input_error)
        self.kernel = self.train_ratio * error
        return self.kernel

    # backconvolution
    def __backconvolution(self, input_error):
        input_size = len(input_error[0])
        data_size = len(self.train_data[0])
        # make output_error array
        c_result_size = data_size
        output_error = np.zeros([self.channel, self.kernel_size, self.kernel_size])
        # back convolution
        for h in range(self.channel):
            for i in range(c_result_size - input_size + 1):
                for j in range(c_result_size - input_size + 1):
                    z = self.diffact(input_error)
                    output_error[h][i][j] = np.sum(self.train_data[h][i:i + input_size, j:j + input_size] @ z[h])
        return output_error
