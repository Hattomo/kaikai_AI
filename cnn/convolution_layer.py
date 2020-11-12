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
                 in_channel,
                 out_channel,
                 ksize,
                 stride=1,
                 pad=0,
                 k_method="xivier",
                 actfunc="relu",
                 train_ratio=0.005):
        self.stride = stride
        self.pad = pad
        (self.actfunc, self.diffact) = (af.relu, af.diffrelu)
        self.kernel = self.__select_w(k_method, in_channel, out_channel, ksize)
        self.train_ratio = train_ratio

    def __select_w(self, k_method, in_channel,out_channel, ksize):
        if k_method == "xivier":
            return csetting.xivier(in_channel, out_channel, ksize)
        elif k_method == "he":
            return csetting.he(in_channel,out_channel, ksize)
        elif k_method == "test":
            return csetting.test(in_channel, out_channel, ksize)
        sys.stdout.write("Error: The kernel method is not found\n")
        sys.exit(1)

    def __padding(self, pad, train_data):
        (channel, height, width) = np.shape(train_data)
        p_result = np.zeros([channel, height+2*pad, width+2*pad])
        for i in range(channel):
            p_result[i][pad:height+pad,pad:width+pad] = train_data[i]
        return p_result

    def forwardpropagation(self, train_data):
        self.train_data = train_data
        padding_data = self.__padding(self.pad, train_data)
        return self.__forwardconvolution(padding_data)

    def __forwardconvolution(self, train_data):
        (in_channel, d_height, d_width) = np.shape(train_data)
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        # check
        if ((d_height-k_height) % self.stride) or ((d_width-k_width) % self.stride):
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        #convolution
        c_result = self.__convolution(train_data, self.kernel)
        return self.actfunc(c_result)

    def backpropagation(self, input_error):
        # back convolution
        self.kernel -= self.train_ratio * self.__backconvolution(input_error)
        error = self.__convolution(self.__padding(1, self.train_data), np.flip(self.kernel))
        return error

    def __backconvolution(self, input_error):
        z = self.diffact(input_error)
        return self.__convolution(self.train_data, z)

    def __convolution(self, mask, _filter):
        (m_channel, m_height, m_width) = np.shape(mask)
        (out_channel, in_channel, f_height, f_width) = np.shape(_filter)
        c_result_channel = m_channel
        c_result_height = (m_height-f_height) // self.stride + 1
        c_result_width = (m_width-f_width) // self.stride + 1
        c_result = np.zeros([c_result_channel, c_result_height, c_result_width])
        for i in range(c_result_channel):
            for j in range(c_result_height):
                for k in range(c_result_width):
                    y = mask[:,j:j+f_height,k:k+f_width] * _filter[i]
                    z = np.sum(y)
                    c_result[i][j][k] = z
        max_ = np.max(c_result)
        return c_result/max_ * 255
                    
