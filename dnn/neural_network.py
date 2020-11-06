import math
import sys

import numpy as np

import activationfunction as af
import costfunction
import files
import dsetting
import vectormath as vmath

class Neural_Network:

    def __init__(self, structure):
        self.structure = structure
        self.layer_num = len(structure)
        self.y = dsetting.ynet(self.structure)
        self.z = dsetting.znet(self.structure)
        self.weight = None
        self.actfunc = None
        self.diffact = None
        self.costfunc = None
        self.diffcost = None
        self.train_ratio = 0.5
        self.cost = list()

    def model(self, w_method="unif", actfunc="sigmoid", costfunc="rss"):
        # 重みの初期化
        if w_method == "xivier":
            self.weight = dsetting.wnet(self.structure, dsetting.xivier)
        elif w_method == "he":
            self.weight = dsetting.wnet(self.structure, dsetting.he)
        elif w_method == "unif":
            self.weight = dsetting.wnet(self.structure, dsetting.unif)
        elif w_method == "load-weight":
            self.weight = files.load(w_method)
        else:
            sys.stdout.write("Error: The weight method is not defined\n")
            sys.exit(1)
        # 活性化関数の初期化
        if actfunc == "sigmoid":
            self.actfunc = af.sigmoid
            self.diffact = af.diffsigmoid
        elif actfunc == "tanh":
            self.actfunc = af.tanh
            self.diffact = af.difftanh
        elif actfunc == "relu":
            self.actfunc = af.vrelu
            self.diffact = af.diffrelu
        elif actfunc == "identity":
            self.actfunc = af.identity
            self.diffact = af.diffidentity
        elif actfunc == "bentIdentity":
            self.actfunc = af.bentIdentity
            self.diffact = af.diffbentIdentity
        elif actfunc == "hardShrink":
            self.actfunc = af.hardShrink
            self.diffact = af.diffhardShrink
        elif actfunc == "log_Sigmoid":
            self.actfunc = af.logSigmoid
            self.diffact = af.difflogSigmoid
        elif actfunc == "tanhShrink":
            self.actfunc = af.tanhShrink
            self.diffact = af.difftanhShrink
        elif actfunc == "elu":
            self.actfunc = af.elu
            self.diffact = af.diffelu
        elif actfunc == "swish":
            self.actfunc = af.swish
            self.diffact = af.diffswish
        elif actfunc == "mish":
            self.actfunc = af.mish
            self.diffact = af.diffmish
        else:
            sys.stdout.write("Error: The actfunc is not found\n")
            sys.exit(1)
        # 損失関数の初期化
        if costfunc == "rss":
            self.costfunc = costfunction.rss
            self.diffcost = costfunction.diffrss
        else:
            sys.stdout.write("Error: The lossfunc is not found\n")
            sys.exit(1)

    # フォワードプロパゲーション
    def forwordpropagation(self, train_data):
        self.z[0][0] = 1
        self.z[0][1:] = train_data
        for i in range(self.layer_num - 1):
            if i == len(self.weight) - 1:
                self.y[i + 1] = self.weight[i] @ self.z[i]
                self.z[i + 1] = self.actfunc(self.y[i + 1])
                break
            self.y[i + 1][1:] = self.weight[i] @ self.z[i]
            self.z[i + 1] = self.actfunc(self.y[i + 1])

    # バックプロパゲーション
    def backpropagation(self, train_data, train_label):
        #学習率の変更
        self.fit_train_ratio(train_label, self.z[-1])
        # out layer to middle layer
        tmp = self.diffact(self.y[-1]) * self.diffcost(train_label, self.z[-1])
        diff = vmath.vvmat(self.z[-2], tmp)
        self.weight[-1] -= self.train_ratio * diff.T
        # middle layer to input layer
        for i in range(self.layer_num - 2):
            weight = (self.weight[-i - 1].T[1:]).T
            z = (self.z[-i - 2].T[1:]).T
            tmp = self.diffact(z) * (weight.T @ tmp)
            diff = vmath.vvmat(self.z[-i - 3], tmp)
            self.weight[-i - 2] -= self.train_ratio * diff.T

    def fit_train_ratio(self, train_label, ans):
        if self.costfunc(train_label, ans) < 0.5:
            self.train_ratio = 0.1
        elif self.costfunc(train_label, ans) < 0.1:
            self.train_ratio = 0.01

    # 学習
    def train(self, train_data, train_label):
        length = len(train_data)
        for i in range(length):
            self.forwordpropagation(train_data[i])
            self.backpropagation(train_data[i], train_label[i])

    # テスト
    def test(self, test_data, test_label):
        count = 0
        cost = 0
        output_node = self.structure[-1]
        z = np.zeros(output_node)
        length = len(test_data)
        for i in range(length):
            self.forwordpropagation(test_data[i])
            for j in range(output_node):
                if self.z[-1][j] >= 0.8:
                    z[j] = 1.0
                elif self.z[-1][j] <= 0.2:
                    z[j] = 0.0
            if (z == test_label[i]).all():
                count += 1
            cost += self.costfunc(test_label[i], self.z[-1])
        self.cost.append(cost / length)
        if math.isnan(cost):
            sys.stdout.write("Error: Due to cost became [nan], Calcuration Stopped\n")
            sys.exit(2)
        print(
            str(count) + "/" + str(length) + " = " + str(count / length) + " : " + str(cost / length) + " : " +
            str(len(self.cost)))
