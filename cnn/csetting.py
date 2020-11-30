# 重みと中間層の作成
import math

import numpy as np

def select_kernel(k_method, in_channel, out_channel, ksize):
    if k_method == "xavier":
        return cnn_xavier(in_channel, out_channel, ksize)
    elif k_method == "he":
        return cnn_he(in_channel, out_channel, ksize)
    elif k_method == "unif":
        return cnn_unif(in_channel, out_channel, ksize)
    sys.stdout.write("Error: The kernel method is not found\n")
    sys.exit(1)
#一様分布
def cnn_unif(in_channel, out_channel, ksize):
    kernel = np.random.rand(in_channel * out_channel * ksize**2) * 2 - 1
    kernel = kernel.reshape([out_channel, in_channel, ksize, ksize])
    return kernel

#正規分布(xavier)
def cnn_xavier(in_channel, out_channel, ksize):
    kernel = np.random.normal(loc=0.0, scale=1 / math.sqrt(out_channel), size=in_channel * out_channel * ksize**2)
    return kernel.reshape(out_channel, in_channel, ksize, ksize)

#正規分布(he)
def cnn_he(in_channel, out_channel, ksize):
    kernel = np.random.normal(loc=0.0, scale=1 / math.sqrt(2 / out_channel), size=in_channel * out_channel * ksize**2)
    return kernel.reshape(out_channel, in_channel, ksize, ksize)

