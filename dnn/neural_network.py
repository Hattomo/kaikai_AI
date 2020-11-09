import math
import sys

import numpy as np

sys.path.append('./shared')
import activationfunction as af
import costfunction
import files
import dsetting
import vectormath as vmath

class Neural_Network:

    def __init__(self, structure, w_method="xivier", actfunc="sigmoid", costfunc="rss"):
        self.structure = structure
        self.y = dsetting.ynet(self.structure)
        self.z = dsetting.znet(self.structure)
        self.weight = self.__set_weight(self.structure, w_method)
        (self.actfunc, self.diffact) = self.__set_actfunc(actfunc)
        (self.costfunc, self.diffcost) = self.__set_costfunc(costfunc)
        self.train_ratio = 0.5
        self.cost = list()

    def __set_weight(self, structure, w_method):
        if w_method == "xivier":
            return dsetting.wnet(structure, dsetting.xivier)
        elif w_method == "he":
            return dsetting.wnet(structure, dsetting.he)
        elif w_method == "unif":
            return dsetting.wnet(structure, dsetting.unif)
        else:
            return files.load(w_method)

    def __set_actfunc(self, actfunc):
        if actfunc == "sigmoid":
            return (af.sigmoid, af.diffsigmoid)
            diffact = af.diffsigmoid
        elif actfunc == "tanh":
            return (af.tanh, af.difftanh)
        elif actfunc == "relu":
            return (af.relu, af.diffrelu)
        elif actfunc == "identity":
            return (af.identity, af.diffidentity)
        elif actfunc == "bentIdentity":
            return (af.bentIdentity, af.diffbentIdentity)
        elif actfunc == "hardShrink":
            return (af.hardShrink, af.diffhardShrink)
        elif actfunc == "log_Sigmoid":
            return (af.logSigmoid, af.difflogSigmoid)
        elif actfunc == "tanhShrink":
            return (af.tanhShrink, af.difftanhShrink)
        elif actfunc == "elu":
            return (af.elu, af.diffelu)
        elif actfunc == "swish":
            return (af.swish, af.diffswish)
        elif actfunc == "mish":
            return (af.mish, af.diffmish)
        sys.stdout.write("Error: The actfunc is not found\n")
        sys.exit(1)

    def __set_costfunc(self, costfunc):
        if costfunc == "rss":
            return (costfunction.rss, costfunction.diffrss)
        sys.stdout.write("Error: The lossfunc is not found\n")
        sys.exit(1)

    def forwordpropagation(self, train_data):
        self.z[0][0] = 1
        self.z[0][1:] = train_data
        for i in range(len(self.structure) - 2):
            self.y[i + 1][1:] = self.weight[i] @ self.z[i]
            self.z[i + 1] = self.actfunc(self.y[i + 1])
        self.y[-1] = self.weight[-1] @ self.z[-2]
        self.z[-1] = self.actfunc(self.y[-1])

    def backpropagation(self, train_data, train_label):
        self.__fit_train_ratio(train_label, self.z[-1])
        # out layer to middle layer
        tmp = self.diffact(self.y[-1]) * self.diffcost(train_label, self.z[-1])
        diff = vmath.vvmat(self.z[-2], tmp)
        self.weight[-1] -= self.train_ratio * diff.T
        # middle layer to input layer
        for i in range(len(self.structure) - 2):
            tmp = self.diffact(self.z[-i - 2][1:]) * (self.weight[-i - 1][:, 1:].T @ tmp)
            diff = vmath.vvmat(self.z[-i - 3], tmp)
            self.weight[-i - 2] -= self.train_ratio * diff.T

    def __fit_train_ratio(self, train_label, ans):
        if self.costfunc(train_label, ans) < 0.5:
            self.train_ratio = 0.1
        elif self.costfunc(train_label, ans) < 0.1:
            self.train_ratio = 0.01

    def train(self, train_data, train_label):
        for i in range(len(train_data)):
            self.forwordpropagation(train_data[i])
            self.backpropagation(train_data[i], train_label[i])

    def test(self, test_data, test_label):
        count = 0
        cost = 0
        length = len(test_data)
        for i in range(length):
            self.forwordpropagation(test_data[i])
            if self.__compare(test_label[i], self.z[-1]):
                count += 1
            cost += self.costfunc(test_label[i], self.z[-1])
        self.cost.append(cost / length)
        if math.isnan(cost):
            sys.stdout.write("Error: Due to cost became [nan], Calcuration Stopped\n")
            sys.exit(2)
        print(
            str(count) + "/" + str(length) + " = " + str(count / length) + " : " + str(cost / length) + " : " +
            str(len(self.cost)))

    def __compare(self, label, predict):
        z = np.zeros(len(label))
        for i in range(len(label)):
            if predict[i] >= 0.8:
                z[i] = 1.0
            elif predict[i] <= 0.2:
                z[i] = 0.0
            else:
                z[i] = predict[i]
        if (z == label).all():
            return True
        return False