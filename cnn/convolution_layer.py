import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import activationfunction as af
import setting

class Convolution_Layer:

    def __init__(self, in_channel, out_channel, ksize, stride=1, pad=0, k_method="xavier", actfunc="relu"):
        self.stride = stride
        self.pad = pad
        (self.actfunc, self.diffact) = (af.relu, af.diffrelu)
        self.kernel = setting.select_kernel(k_method, in_channel, out_channel, ksize)
        self.move = [[], [], []]

    def __padding(self, pad, train_data):
        (batch, channel, height, width) = np.shape(train_data)
        # <zero padding>
        p_result = np.zeros([batch, channel, height + 2*pad, width + 2*pad])
        for i in range(batch):
            p_result[i][:, pad:height + pad, pad:width + pad] = train_data[i]
        return p_result

    def forwardpropagation(self, image_data):
        (batch, in_channel, img_height, img_width) = np.shape(image_data)
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        # to prevent error from setting wrong stride
        if ((img_height-k_height) % self.stride) or ((img_width-k_width) % self.stride):
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        # <padding>  Be careful,size of train data change!
        # (padding size is [channel, height+2*pad, width+2*pad])
        self.train_data = np.ones([batch, in_channel, img_height + 2 * self.pad, img_width + 2 * self.pad])
        self.train_data[:, 1:] = self.__padding(self.pad, image_data)
        # <convolution>
        c_result = self.convolution(self.train_data, self.kernel)
        c_result = self.actfunc(c_result)
        # rescaling method
        # function(c_result)
        return c_result

    def backpropagation(self, input_error):
        train_ratio = 0.01
        (batch, in_channel, tr_height, tr_width) = np.shape(self.train_data)
        (batch, out_channel, er_height, er_width) = np.shape(input_error)
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        # update kernel
        result = np.zeros([out_channel, in_channel, k_height, k_width])
        kernelmove = 0
        for h in range(batch):
            for i in range(out_channel):
                for j in range((tr_height-er_height) // self.stride + 1):
                    for k in range((tr_width-er_width) // self.stride + 1):
                        y = self.train_data[h][:, j:j + er_height, k:k + er_width] * input_error[h][i]
                        result[i][:, j, k] = np.sum(y, axis=(1, 2))
            self.kernel -= train_ratio * result
            kernelmove += np.sum(abs(train_ratio * result))
        self.move[0].append(np.max(self.kernel))
        self.move[1].append(np.min(self.kernel))
        self.move[2].append(kernelmove)

        # make next error
        (out_channel, in_channel, k_height, k_width) = np.shape(self.kernel)
        _kernel = np.zeros([in_channel, out_channel, k_height, k_width])
        for i in range(out_channel):
            for j in range(in_channel):
                _kernel[j][i] = np.flip(self.kernel[i][j])
        error = self.convolution(self.__padding(1, input_error), _kernel)
        #error = self.diffact(error)
        return error

    def convolution(self, image, kernel):
        (batch, img_channel, img_height, img_width) = np.shape(image)
        (out_channel, in_channel, k_height, k_width) = np.shape(kernel)
        # the number of stride
        stride_height = (img_height-k_height) // self.stride + 1
        stride_width = (img_height-k_height) // self.stride + 1
        # make matrix
        col_size = [batch, img_channel, k_height, k_width, stride_height, stride_width]
        col = self.im2col(image, col_size)
        flat_kernel = kernel.flatten().reshape(img_channel * k_height * k_width, -1, order="F")
        # make output image size
        channel = out_channel
        height = (img_height-k_height) // self.stride + 1
        width = (img_width-k_width) // self.stride + 1
        out_size = [batch, channel, height, width]
        # convolution
        result = col @ flat_kernel
        return self.col2im(result, out_size)

    def im2col(self, image, size):
        batch, img_channel, k_height, k_width, stride_height, stride_width = size
        col = np.zeros(size)
        for i in range(k_height):
            i_max = i + self.stride * stride_height
            for j in range(k_width):
                j_max = j + self.stride * stride_width
                col[:, :, i, j, :, :] = image[:, :, i:i_max:self.stride, j:j_max:self.stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * stride_height * stride_width, -1)
        return col

    def col2im(self, col, size):
        batch, channel, height, width = size
        return col.T.flatten().reshape(batch, channel, height, width)
