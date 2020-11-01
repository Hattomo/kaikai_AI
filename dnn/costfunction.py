import math

# Residual Sum-Of-Squares
def rss(ansdata, data):
    return (data - ansdata)**2

# diff of Residual Sum-Of-Squares
def diffrss(ansdata, data):
    return data - ansdata
