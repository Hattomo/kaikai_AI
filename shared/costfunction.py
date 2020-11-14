import math
import numpy as np

def rss(label, ans):
    return np.sum((ans - label)**2)

def diffrss(label, ans):
    return ans - label

def cross_entropy(label, ans):
    sum_ = 0
    delta = 1e-5
    if label.size == 1:
        return -label * math.log(abs(ans) + delta)
    for i in range(label.size):
        if ans <= 0:
            return label.size
        sum_ += -label[i] * math.log(abs(ans) + delta)
    return sum_

def diffcross_entropy(label, ans):
    diff = ans
    delta = 1e-5
    for i in range(label.size):
        diff[i] = -label[i] / (ans+delta)
    return diff