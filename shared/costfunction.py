import math

# Residual Sum-Of-Squares
def rss(label, ans):
    sum_ = 0
    if len(label) == 1:
        return (ans - label)**2
    for i in range(len(label)):
        sum_ += (ans[i] - label[i])**2
    return sum_

# diff of Residual Sum-Of-Squares
def diffrss(label, ans):
    return ans - label

def cross_entropy(label, ans):
    sum_ = 0
    delta = 1e-5
    if label.size == 1:
        return -label * math.log(abs(ans)+delta)
    for i in range(label.size):
        if ans <= 0:
            return label.size
        sum_ += -label[i] * math.log(abs(ans)+delta)
    return sum_

def diffcross_entropy(label, ans):
    diff = ans
    delta = 1e-5
    for i in range(label.size):
        diff[i] = -label[i] / (ans+delta)
    return diff