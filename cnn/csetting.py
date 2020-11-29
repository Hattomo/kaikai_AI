# 重みと中間層の作成
import math
import random

import numpy as np

# 重みの作成
#一様分布
def unif(in_channel, out_channel, ksize):
    kernel = np.random.rand(in_channel * out_channel * ksize**2) * 2 - 1
    kernel = kernel.reshape([out_channel, in_channel, ksize, ksize])
    return kernel

#正規分布(xavier)
def xavier(in_channel, out_channel, ksize):
    kernel = np.random.normal(loc=0.0, scale=1 / math.sqrt(out_channel), size=in_channel * out_channel * ksize**2)
    return kernel.reshape(out_channel, in_channel, ksize, ksize)

#正規分布(he)
def he(in_channel, out_channel, ksize):
    kernel = np.random.normal(loc=0.0, scale=1 / math.sqrt(2 / out_channel), size=in_channel * out_channel * ksize**2)
    return kernel.reshape(out_channel, in_channel, ksize, ksize)

def test(in_channel, out_channel, ksize):
    kernel = np.zeros(out_channel * in_channel * ksize**2)
    for i in range(out_channel * in_channel * ksize**2):
        kernel[i] = (i+1) / 10
    return kernel.reshape([out_channel, in_channel, ksize, ksize])
