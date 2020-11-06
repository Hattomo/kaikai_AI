import sys

import numpy as np

sys.path.append('./dnn')
import neural_network as nn
import activationfunction as af
import setting

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

    def __call__(self, data):
        self.data = data
        self.padding()
        self.convolution()

    def __select_w(self, k_method):
        if k_method == "xivier":
            self.kernel = setting.knet(self.channel, self.kernel_size, setting.xivier)
        elif k_method == "he":
            self.kernel = setting.knet(self.channel, self.kernel_size, setting.he)
        else:
            sys.stdout.write("Error: The kernel method is not found\n")
            sys.exit(1)

    def padding(self):
        if self.padding_method == "valid-padding":
            x = self.data
        elif self.padding_method == "same-padding":
            pass
        else:
            sys.stdout.write("Error: The padding method is not found\n")
            sys.exit(1)

    def convolution(self):
        data_hight = len(self.data)
        data_width = len(self.data[0])
        if (data_hight - self.kernel_size) % self.stride == 0 and (data_width - self.kernel_size) % self.stride == 0:
            out_hight = int((data_hight - self.kernel_size) / self.stride + 1)
            out_width = int((data_width - self.kernel_size) / self.stride + 1)
            self.out = np.zeros([self.channel, out_hight, out_width])
        else:
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        #convolution
        for h in range(self.channel):
            for i in range(out_hight):
                for j in range(out_width):
                    y = (self.data[i:i + self.kernel_size].T[j:j + self.kernel_size].T) @ self.kernel[h]
                    z = af.mrelu(y)
                    self.out[h][i][j] = np.sum(z)
