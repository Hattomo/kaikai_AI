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
