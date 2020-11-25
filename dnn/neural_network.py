import math
import sys

import numpy as np

sys.path.append('./shared')
import activationfunction as af
import costfunction as cf
import dsetting
import numpy_files as npfiles
import optimaize_method as om
import vectormath as vmath

class Neural_Network:

    def __init__(self,
                 structure,
                 dropout=[0, 0, 0],
                 w_method="xivier",
                 actfunc="sigmoid",
                 costfunc="rss",
                 testmode="classify",
                 optimaize_method="adam"):
        self.structure = structure
        self.dropout = dropout
        self.testmode = testmode
        self.z, self.y = dsetting.set_layer(self.structure)
        self.do = dsetting.donet(self.structure)
        self.weight = dsetting.set_weight(self.structure, w_method)
        self.actfunc, self.diffact = af.set_actfunc(actfunc)
        self.costfunc, self.diffcost = cf.set_costfunc(costfunc)
        self.optimaize_method = self.set_optimaize_method(optimaize_method)
        self.optimaize_method_name = optimaize_method
        self.train_ratio = 0.1
        self.cost = list()
        self.accurancy = []

    def set_optimaize_method(self, optimaize_method):
        if (optimaize_method == "gd"):
            return om.GradientDesend()
        elif (optimaize_method == "adam"):
            return om.Adam(self.structure)
        else:
            sys.stdout.write("Error: The optimaize_method is not found\n")
            sys.exit(1)

    def optimaize(self, error, cnt, layer):
        if (self.optimaize_method_name == "gd"):
            if (len(self.cost) == 0):
                cost = 1000
            else:
                cost = self.cost[-1]
            return self.optimaize_method.optimaize(error, cost)
        elif (self.optimaize_method_name == "adam"):
            return self.optimaize_method.optimaize(error, cnt + 1, layer)

    def forwardpropagation(self, train_data, istrain=True):
        self.z[0][0] = 1
        self.z[0][1:] = train_data
        for i in range(len(self.structure) - 2):
            self.y[i + 1][1:] = self.weight[i] @ self.do[i] @ self.z[i]
            if istrain:
                self.z[i + 1] = self.actfunc(self.y[i + 1])
            else:
                self.z[i + 1] = self.actfunc(self.y[i + 1]) * (1 - self.dropout[i])
        self.y[-1] = self.weight[-1] @ self.do[-1] @ self.z[-2]
        if istrain:
            self.z[-1] = self.actfunc(self.y[-1])
        else:
            self.z[-1] = self.actfunc(self.y[-1]) * (1 - self.dropout[-1])

    # バックプロパゲーション
    def backpropagation(self, train_data, train_label, cnt, isexternal=False):
        # out layer to middle layer
        tmp = self.diffact(self.y[-1]) * self.diffcost(train_label, self.z[-1])
        diff = vmath.vvmat(self.z[-2], tmp)
        self.weight[-1] -= self.optimaize(diff.T, cnt, len(self.structure) - 1 - 1) @ self.do[-1]
        # middle layer to input layer
        for i in range(len(self.structure) - 2):
            tmp = self.diffact(self.z[-i - 2][1:]) * ((self.weight[-i - 1][:, 1:] @ self.do[-i - 1][:-1, :-1]).T @ tmp)
            diff = vmath.vvmat(self.z[-i - 3], tmp)
            self.weight[-i - 2] -= self.optimaize(diff.T, cnt, len(self.structure) - 3 - i)
        if isexternal:
            weight = self.weight[0].T[1:]
            return weight @ tmp

    def __dropout_shake(self, istrain=True):
        for i in range(len(self.structure) - 1):
            for j in range(self.do[i].shape[1]):
                if istrain:
                    if (np.random.rand() < self.dropout[i]):
                        self.do[i][j][j] = 0
                else:
                    shape = np.shape(self.do[i])
                    self.do[i] = np.identity(shape[0])

    # 学習
    def train(self, train_data, train_label, isexternal=False):
        self.__dropout_shake()
        for i in range(len(train_data)):
            self.forwardpropagation(train_data[i], False)
            self.backpropagation(train_data[i], train_label[i], i, isexternal)

    def test(self, test_data, test_label):
        self.__dropout_shake(False)
        count = 0
        cost = 0
        length = len(test_data)
        for i in range(length):
            self.forwardpropagation(test_data[i], False)
            if (abs(test_label[i] - self.z[-1]) < 0.2).all():
                # if self.__compare(test_label[i], self.z[-1]):
                count += 1
            cost += self.costfunc(test_label[i], self.z[-1])
        self.cost.append(cost / length)
        self.accurancy.append(count / length)
        if math.isnan(cost):
            sys.stdout.write("Error: Due to cost became [nan], Calcuration Stopped\n")
            sys.exit(2)
        print(
            str(count) + "/" + str(length) + " = " + str(count / length) + " : " + str(cost / length) + " : " +
            str(len(self.cost)))

    def __compare(self, label, predict):
        if self.testmode == "classify":
            if np.argmax(predict) == np.argmax(label):
                return True
            return False
        elif self.testmode == "predict":
            sys.stdout.write("Error : __compare() mode predict is not supported")
            sys.exit(7)
        else:
            sys.stdout.write("Error : __compare() mode is not right")
            sys.exit(6)
