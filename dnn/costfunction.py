import math

# Residual Sum-Of-Squares
def rss(ansdata, data):
    sum_ = 0
    if len(ansdata) == 1:
        return (data - ansdata)**2
    for i in range(len(ansdata)):
        sum_ += (data[i] - ansdata[i])**2
    return sum_

# diff of Residual Sum-Of-Squares
def diffrss(ansdata, data):
    return data - ansdata
