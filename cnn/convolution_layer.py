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
                 train_ratio=0.001):
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
        (batch, channel, height, width) = np.shape(train_data)
        # <zero padding>
        p_result = np.zeros([batch, channel, height + 2*pad, width + 2*pad])
        for i in range(batch):
            p_result[i][:, pad:height + pad, pad:width + pad] = train_data[i]
        return p_result

    def forwardpropagation(self, image_data):
        self.image_data = image_data
        (batch, in_channel, img_height, img_width) = np.shape(image_data)
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        # to prevent error from setting wrong stride
        if ((img_height-k_height) % self.stride) or ((img_width-k_width) % self.stride):
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        # <padding>  Be careful,size of train data change!
        # (padding size is [channel, height+2*pad-k_height/stride, width+2*pad-k_height/stride])
        self.train_data = self.__padding(self.pad, image_data)
        # <convolution>
        c_result = self.__convolution(self.train_data, self.kernel)
        c_result = self.actfunc(c_result)
        # rescaling method
        # function(c_result)
        return c_result

    def backpropagation(self, input_error):
        (batch, in_channel, tr_height, tr_width) = np.shape(self.train_data)
        (batch, out_channel, er_height, er_width) = np.shape(input_error)
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        # update kernel
        result = np.zeros([out_channel, in_channel, k_height, k_width])
        for h in range(batch):
            for i in range(out_channel):
                for j in range((tr_height-er_height) // self.stride + 1):
                    for k in range((tr_width-er_width) // self.stride + 1):
                        y = self.train_data[h][:, j:j + er_height, k:k + er_width] * input_error[h][i]
                        result[i][:, j, k] = np.sum(y, axis=(1, 2))
            self.kernel -= self.train_ratio * result
        # make next error
        # make next error
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        _kernel = np.zeros([in_channel, out_channel, k_height, k_width])
        for i in range(out_channel):
            for j in range(in_channel):
                _kernel[j][i] = np.flip(self.kernel[i][j])
        error = self.__convolution(self.__padding(1, input_error), _kernel)
        #error = self.diffact(error)
        # return error
        return error
        # _kernel = np.flip(self.kernel, axis=(2, 3))
        # error = self.__convolution(self.__padding(1, input_error), _kernel)
        # return error

    def __convolution(self, mask, _filter):
        (batch, m_channel, m_height, m_width) = np.shape(mask)
        (out_channel, in_channel, f_height, f_width) = np.shape(_filter)
        # make convolution result
        result_channel, result_height, result_width = out_channel, (m_height-f_height) // self.stride + 1, (
            m_width-f_width) // self.stride + 1
        c_result = np.zeros([batch, result_channel, result_height, result_width])
        for h in range(batch):
            for i in range(result_channel):
                for j in range(result_height):
                    for k in range(result_width):
                        y = mask[h][:, j:j + f_height, k:k + f_width] * _filter[i]
                        z = np.sum(y)
                        c_result[h][i][j][k] = z
        return c_result
