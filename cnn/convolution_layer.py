import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import neural_network as nn
import activationfunction as af
import csetting

class Convolution_Layer:

    def __init__(self,
                 channel,
                 kernel_size,
                 k_method="xivier",
                 stride=1,
                 actfunc="sigmoid",
                 padding_method="valid-padding"):
        self.channel = channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.actfunc = actfunc
        self.padding_method = padding_method
        self.__select_w(k_method)

    def convolution(self, train_data):
        padding_data = self.__padding(train_data)
        return self.__convolution(padding_data)

    def backconvolution(self, error):
        pass

    def __select_w(self, k_method):
        if k_method == "xivier":
            self.kernel = csetting.knet(self.channel, self.kernel_size, csetting.xivier)
        elif k_method == "he":
            self.kernel = csetting.knet(self.channel, self.kernel_size, csetting.he)
        else:
            sys.stdout.write("Error: The kernel method is not found\n")
            sys.exit(1)

    def __padding(self, train_data):
        #padding
        if self.padding_method == "valid-padding":
            x = train_data
        else:
            sys.stdout.write("Error: The padding method is not found\n")
            sys.exit(1)
        return x

    def __convolution(self, train_data):
        data_size = len(train_data)
        # make output array
        if (data_size - self.kernel_size) % self.stride == 0:
            out_size = int((data_size - self.kernel_size) / self.stride + 1)
            out = np.zeros([self.channel, out_size, out_size])
        else:
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        #convolution
        for h in range(self.channel):
            for i in range(out_size):
                for j in range(out_size):
                    y = (train_data[i:i + self.kernel_size].T[j:j + self.kernel_size].T) @ self.kernel[h]
                    z = af.sigmoid(y)
                    out[h][i][j] = np.sum(z)
        return out