import numpy as np

class Convolutional_Neural_Network():

    def __init__(self, structure):
        self.structure = structure

    def forwardpropagation(self, train_img, train_label, istrain=True):
        train_data = train_img
        for i in range(len(self.structure) - 1):
            train_data = self.structure[i].forwardpropagation(train_data)
        error = self.structure[-1].train(train_data, train_label)
        if istrain:
            return error
        return train_data

    def reset_all(self):
        for i in range(len(self.structure)):
            self.structure[i].reset()

    def backpropagation(self, error):
        for i in range(len(self.structure) - 1, 0):
            error = self.structure[i].backpropagation(error)

    def train(self, train_img, train_label):
        num, batch, channel, tr_height, tr_width = train_img.shape
        for i in range(num):
            error = self.forwardpropagation(train_img[i], train_label[i])
            self.backpropagation(error)

    def test(self, test_img, test_label, mode="abs"):
        testsize, channel, ts_height, ts_width = test_img.shape
        test_data = self.forwardpropagation(test_img, test_label, False)
        self.structure[-1].test(test_data, test_label, mode)
