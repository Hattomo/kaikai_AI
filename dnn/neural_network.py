import sys
import math
import activationfunction as af
import costfunction
import setting
import numpy as np
import vectormath as vmath
import files

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
        elif actfunc == "ReLU":
            self.actfunc = af.mReLU
            self.difffunc = af.diffReLU
        elif actfunc == "Identity":
            self.actfunc = af.Identity
            self.difffunc = af.diffIdentity
        elif actfunc == "BentIdentity":
            self.actfunc = af.BentIdentity
            self.difffunc = af.diffBentIdentity
        elif actfunc == "hardShrink":
            self.actfunc = af.hardShrink
            self.difffunc = af.diffhardShrink
        elif actfunc == "logSigmoid":
            self.actfunc = af.logSigmoid
            self.difffunc = af.difflogSigmoid
        elif actfunc == "tanhShrink":
            self.actfunc = af.tanhShrink
            self.difffunc = af.difftanhShrink
        elif actfunc == "ELU":
            self.actfunc = af.ELU
            self.difffunc = af.diffELU
        elif actfunc == "Swish":
            self.actfunc = af.Swish
            self.difffunc = af.diffSwish
        elif actfunc == "Mish":
            self.actfunc = af.Mish
            self.difffunc = af.diffMish
        else:
            sys.stdout.write("Error: The actfunc is not found\n")
            sys.exit(1)
        # 損失関数の初期化
        if costfunc == "rss":
            self.costfunc = costfunction.rss
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
        if (self.z[-1] - y)**2 < 0.5:
            self.train_ratio = 0.1
        elif (self.z[-1] - y)**2 < 0.1:
            self.train_ratio = 0.01
        tmp = self.difffunc(self.y[-1]) * (self.z[-1] - y)
        diff = self.z[-2] * tmp
        self.weight[-1] -= self.train_ratio * diff
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
            self.forwordpropagation(self.data[i][:-1])
            self.backpropagation(self.data[i][:-1], self.data[i][-1])

    # テスト
    def test(self):
        count = 0
        cost = 0
        length = len(self.testdata)
        for i in range(length):
            self.forwordpropagation(self.testdata[i][:-1])
            if self.z[-1] >= 0.8 and self.testdata[i][-1] == 1:
                count += 1
            elif self.z[-1] <= 0.2 and self.testdata[i][-1] == 0:
                count += 1
            cost += ((self.z[-1] - self.testdata[i][-1])**2)
        self.cost.append(cost / length)
        if math.isnan(cost):
            sys.stdout.write("Error: Due to cost became [nan], Calcuration Stopped\n")
            sys.exit(2)
        print(
            str(count) + "/" + str(length) + " = " + str(count / length) + " : " + str(cost) + " : " +
            str(len(self.cost)))
