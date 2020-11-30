import math
import sys

import numpy as np

sys.path.append('./shared')
import activationfunction as af
import costfunction as cf
import dsetting

class Neural_Network:

    def __init__(
        self,
        structure,
        dropout=[0, 0, 0],
        w_method="xavier",
        actfunc="sigmoid",
        costfunc="rss",
    ):
        self.structure = structure
        self.dropout = dropout
        self.do = dsetting.donet(self.structure)
        self.weight = dsetting.set_weight(self.structure, w_method)
        self.actfunc, self.diffact = af.set_actfunc(actfunc)
        self.costfunc, self.diffcost = cf.set_costfunc(costfunc)
        self.cost, self.accurancy = list(), list()

    def forwardpropagation(self, train_data, batch):
        self.z, self.y = dsetting.set_layer(self.structure, batch)
        self.z[0][:, 1:] = train_data
        for i in range(len(self.structure) - 2):
            self.y[i + 1][:, 1:] = self.z[i] @ (self.weight[i] @ self.do[i]).T
            self.z[i + 1] = self.actfunc(self.y[i + 1]) * (1 - self.dropout[i])
        self.y[-1] = self.z[-2] @ (self.weight[-1] @ self.do[-1]).T
        self.z[-1] = self.actfunc(self.y[-1]) * (1 - self.dropout[-1])

    def backpropagation(self, train_data, train_label):
        #学習率の変更
        train_ratio = self.__fit_train_ratio(train_label, self.z[-1])
        # out layer to middle layer
        tmp = self.diffact(self.y[-1]) * self.diffcost(train_label, self.z[-1])
        self.weight[-1] -= train_ratio * (tmp.T @ self.z[-2] @ self.do[-1].T)
        # middle layer to input layer
        for i in range(len(self.structure) - 2):
            tmp = self.diffact(self.z[-i - 2][:, 1:]) * (tmp @ self.weight[-i - 1][:, 1:] @ self.do[-i - 1][:-1, :-1].T)
            self.weight[-i - 2] -= train_ratio * tmp.T @ self.z[-i - 3]
        return tmp @ self.weight[0][:, 1:]

    # ドロップアウトするノードの選択
    def __dropout_shake(self, istrain=True):
        for i in range(len(self.structure) - 1):
            self.do[i] = np.eye(len(self.do[i]))
            for j in range(self.do[i].shape[1]):
                if istrain and (np.random.rand() < self.dropout[i]):
                    self.do[i][j][j] = 0

    def __fit_train_ratio(self, train_label, ans):
        cost = self.costfunc(train_label, ans)
        if cost < 0.01:
            return 0.01
        elif cost < 0.1:
            return 0.05
        return 0.1

    def train(self, train_data, train_label,dnn=True):
        if dnn:
            num, batch, data_num = np.shape(train_data)
            for i in range(num):
                self.__dropout_shake()
                self.forwardpropagation(train_data[i], batch)
                self.backpropagation(train_data[i], train_label[i])
        else:
            batch, data_num = np.shape(train_data)
            self.__dropout_shake()
            self.forwardpropagation(train_data[i], batch)
            self.backpropagation(train_data[i], train_label[i])

    def test(self, test_data, test_label, mode="classify"):
        self.__dropout_shake(False)
        count, cost = 0, 0
        length = len(test_data)
        # reset dropout
        dropout = self.dropout
        self.dropout = np.zeros_like(dropout)
        self.forwardpropagation(test_data, length)
        for i in range(length):
            if (abs(test_label[i] - self.z[-1][i]) < 0.2).all():
                # if self.__compare(test_label[i], self.z[-1],mode):
                count += 1
            cost += self.costfunc(test_label[i], self.z[-1][i])
        # set dropout
        self.dropout = dropout
        # add data
        self.cost.append(cost / length)
        self.accurancy.append(count / length)
        if math.isnan(cost):
            sys.stdout.write("Error: Due to cost became [nan], Calcuration Stopped\n")
            sys.exit(2)
        print(
            str(count) + "/" + str(length) + " = " + str(count / length) + " : " + str(cost / length) + " : " +
            str(len(self.cost)))

    def __compare(self, label, predict, mode):
        if mode == "classify":
            if np.argmax(predict) == np.argmax(label):
                return True
            return False
        elif mode == "predict":
            sys.stdout.write("Error : __compare() mode predict is not supported")
            sys.exit(7)
        else:
            sys.stdout.write("Error : __compare() mode is not right")
            sys.exit(6)
