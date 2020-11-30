import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import analysistool as atool
import cnn_analysistool as catool
import convolution_layer as cl
import csetting
import fully_connenct_layer as fc
import neural_network as nn
import normalization_layer as nl
import pooling_layer as pl
import mnist
import dataset

class Convolution_Neural_Network():
    def __init__(self,structure):
        self.structure = structure
    def forwardpropagation(self,train_img,train_label,istrain=True):
        train_data = train_img
        for i in range(len(self.structure)-1):
            train_data = self.structure[i].forwardpropagation(train_data)
        error = self.structure[-1].train(train_data,train_label)
        if istrain:
            return error
        return train_data
    def backpropagation(self,error):
        for i in range(len(self.structure)-1,0):
            error = self.structure[i].backpropagation(error)
    def train(self,train_img,train_label):
        num, batch, channel, tr_height, tr_width = train_img.shape
        for i in range(num):
            error = self.forwardpropagation(train_img[i],train_label[i])
            self.backpropagation(error)
    def test(self,test_img,test_label):
        testsize, channel, ts_height, ts_width = test_img.shape
        test_data = self.forwardpropagation(test_img,test_label,False)
        self.structure[-1].test(test_data,test_label)