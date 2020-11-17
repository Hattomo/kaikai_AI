import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import activationfunction as af
import csetting

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

    def __select_w(self, k_method, in_channel, out_channel, ksize):
        if k_method == "xivier":
            return csetting.xivier(in_channel, out_channel, ksize)
        elif k_method == "he":
            return csetting.he(in_channel, out_channel, ksize)
        elif k_method == "test":
            return csetting.test(in_channel, out_channel, ksize)
        sys.stdout.write("Error: The kernel method is not found\n")
        sys.exit(1)

    def __padding(self, pad, train_data):
        (channel, height, width) = np.shape(train_data)
        # <zero padding>
        p_result = np.zeros([channel, height + 2*pad, width + 2*pad])
        for i in range(channel):
            p_result[i][pad:height + pad, pad:width + pad] = train_data[i]
        return p_result

    def forwardpropagation(self, image_data):
        self.image_data = image_data
        (in_channel, img_height, img_width) = np.shape(image_data)
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        # to prevent error from setting wrong stride
        if ((d_height-k_height) % self.stride[0]) or ((d_width-k_width) % self.stride[1]):
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        # <padding>  Be careful,size of train data change!
        # (padding size is [channel, height+2*pad-k_height/stride, width+2*pad-k_height/stride])
        padding_data = self.__padding(self.pad, train_data)
        # <convolution>
        c_result = self.__convolution(padding_data, self.kernel)
        return self.actfunc(c_result)

    def backpropagation(self, input_error):
        # updata kernel
        z = self.diffact(input_error)
        b_result = self.__convolution(self.train_data, z)
        self.kernel -= self.train_ratio * b_result
        # make next error
        error = self.__convolution(self.__padding(1, self.train_data), np.flip(self.kernel))
        return error

    def __convolution(self, mask, _filter):
        (m_channel, m_height, m_width) = np.shape(mask)
        (out_channel, in_channel, f_height, f_width) = np.shape(_filter)
        # make convolution result
        c_result_channel = m_channel
        c_result_height = (m_height-f_height) // self.stride + 1
        c_result_width = (m_width-f_width) // self.stride + 1
        c_result = np.zeros([c_result_channel, c_result_height, c_result_width])
        for i in range(c_result_channel):
            for j in range(c_result_height):
                for k in range(c_result_width):
                    y = mask[:, j:j + f_height, k:k + f_width] * _filter[i]
                    z = np.sum(y)
                    c_result[i][j][k] = z
        # The feature(output_data) have to have values from 0 to 255
        c_result = c_result / np.max(c_result) * 255
        return c_result
