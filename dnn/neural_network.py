import math
import sys

import numpy as np

import activationfunction as af
import costfunction
import files
import setting
import vectormath as vmath

class Neural_Network:

    def __init__(self, layer):
        self.layer = layer
        self.length = len(layer)
        self.y = None
        self.z = None
        self.weight = None
        self.backweight = None
        self.actfunc = None
        self.difffunc = None
        self.costfunc = None
        self.diffcost = None
        self.train_ratio = 0.5
        self.cost = list()

    def model(self, data, testdata, w_method="unif", actfunc="sigmoid", costfunc="rss"):
        self.data = data
        self.testdata = testdata
        self.y = setting.ynet(self.layer)
        self.z = setting.znet(self.layer)
        # 重みの初期化
        if w_method == "xivier":
            self.weight = setting.wnet(self.layer, setting.xivier)
        elif w_method == "he":
            self.weight = setting.wnet(self.layer, setting.he)
        elif w_method == "unif":
            self.weight = setting.wnet(self.layer, setting.unif)
        else:
            self.weight = files.load(w_method)
        # 活性化関数の初期化
        if actfunc == "sigmoid":
            self.actfunc = af.sigmoid
            self.difffunc = af.diffsigmoid
        elif actfunc == "tanh":
            self.actfunc = af.tanh
            self.difffunc = af.difftanh
        elif actfunc == "relu":
            self.actfunc = af.mrelu
            self.difffunc = af.diffrelu
        elif actfunc == "identity":
            self.actfunc = af.identity
            self.difffunc = af.diffidentity
        elif actfunc == "bentIdentity":
            self.actfunc = af.bentIdentity
            self.difffunc = af.diffbentIdentity
        elif actfunc == "hardShrink":
            self.actfunc = af.hardShrink
            self.difffunc = af.diffhardShrink
        elif actfunc == "log_Sigmoid":
            self.actfunc = af.logSigmoid
            self.difffunc = af.difflogSigmoid
        elif actfunc == "tanhShrink":
            self.actfunc = af.tanhShrink
            self.difffunc = af.difftanhShrink
        elif actfunc == "elu":
            self.actfunc = af.elu
            self.difffunc = af.diffelu
        elif actfunc == "swish":
            self.actfunc = af.swish
            self.difffunc = af.diffswish
        elif actfunc == "mish":
            self.actfunc = af.mish
            self.difffunc = af.diffmish
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
        self.difffunc = af.msigmoid

    # フォワードプロパゲーション
    def forwordpropagation(self, x):
        self.z[0][0] = 1
        self.z[0][1:] = x
        for i in range(len(self.layer) - 1):
            if i == len(self.weight) - 1:
                self.y[i + 1] = self.weight[i] @ self.z[i]
                self.z[i + 1] = self.actfunc(self.y[i + 1])
                break
            self.y[i + 1][1:] = self.weight[i] @ self.z[i]
            self.z[i + 1] = self.actfunc(self.y[i + 1])

    # バックプロパゲーション
    def backpropagation(self, x, y):
        # out layer to middle layer
        if self.costfunc(y, self.z[-1]) < 0.5:
            self.train_ratio = 0.1
        elif self.costfunc(y, self.z[-1]) < 0.1:
            self.train_ratio = 0.01
        tmp = self.difffunc(self.y[-1]) * self.diffcost(y, self.z[-1])
        diff = vmath.vvmat(self.z[-2], tmp)
        self.weight[-1] -= self.train_ratio * diff.T
        # middle layer to input layer
        for i in range(len(self.layer) - 2):
            weight = (self.weight[-i - 1].T[1:]).T
            z = (self.z[-i - 2].T[1:]).T
            tmp = self.difffunc(z) * (weight.T @ tmp)
            diff = vmath.vvmat(self.z[-i - 3], tmp)
            self.weight[-i - 2] -= self.train_ratio * diff.T

    # 学習
    def train(self):
        length = len(self.data)
        for i in range(length):
            self.forwordpropagation(self.data[i][:-self.layer[-1]])
            self.backpropagation(self.data[i][:-self.layer[-1]], self.data[i][-self.layer[-1]:])

    # テスト
    def test(self):
        count = 0
        cost = 0
        z = np.zeros(self.layer[-1], dtype=np.float128)
        length = len(self.testdata)
        for i in range(length):
            self.forwordpropagation(self.testdata[i][:-self.layer[-1]])
            for j in range(self.layer[-1]):
                if self.z[-1][j] >= 0.8:
                    z[j] = 1.0
                elif self.z[-1][j] <= 0.2:
                    z[j] = 0.0
            if (z == self.testdata[i][-self.layer[-1]:]).all():
                count += 1
            cost += self.costfunc(self.testdata[i][-self.layer[-1]:], self.z[-1])
        self.cost.append(cost / length)
        if math.isnan(cost):
            sys.stdout.write("Error: Due to cost became [nan], Calcuration Stopped\n")
            sys.exit(2)
        print(
            str(count) + "/" + str(length) + " = " + str(count / length) + " : " + str(cost / length) + " : " +
            str(len(self.cost)))
