import math
import sys

import numpy as np

sys.path.append('./shared')
import activationfunction as af
import costfunction
import numpy_files as npfiles
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

    # 重みの初期化
    def __set_weight(self, structure, w_method):
        if w_method == "xivier":
            weight = dsetting.wnet(structure, dsetting.xivier)
        elif w_method == "he":
            weight = dsetting.wnet(structure, dsetting.he)
        elif w_method == "unif":
            weight = dsetting.wnet(structure, dsetting.unif)
        else:
            weight = npfiles.load(w_method)
        return weight

    # 活性化関数の初期化
    def __set_actfunc(self, actfunc):
        if actfunc == "sigmoid":
            func = af.sigmoid
            diffact = af.diffsigmoid
        elif actfunc == "tanh":
            func = af.tanh
            diffact = af.difftanh
        elif actfunc == "relu":
            func = af.vrelu
            diffact = af.diffrelu
        elif actfunc == "identity":
            func = af.identity
            diffact = af.diffidentity
        elif actfunc == "bentIdentity":
            func = af.bentIdentity
            diffact = af.diffbentIdentity
        elif actfunc == "hardShrink":
            func = af.hardShrink
            diffact = af.diffhardShrink
        elif actfunc == "log_Sigmoid":
            func = af.logSigmoid
            diffact = af.difflogSigmoid
        elif actfunc == "tanhShrink":
            func = af.tanhShrink
            diffact = af.difftanhShrink
        elif actfunc == "elu":
            func = af.elu
            diffact = af.diffelu
        elif actfunc == "swish":
            func = af.swish
            diffact = af.diffswish
        elif actfunc == "mish":
            func = af.mish
            diffact = af.diffmish
        else:
            sys.stdout.write("Error: The actfunc is not found\n")
            sys.exit(1)
        return (func, diffact)

    # 損失関数の初期化
    def __set_costfunc(self, costfunc):
        if costfunc == "rss":
            func = costfunction.rss
            diffcost = costfunction.diffrss
        else:
            sys.stdout.write("Error: The lossfunc is not found\n")
            sys.exit(1)
        return (func, diffcost)

    # フォワードプロパゲーション
    def forwordpropagation(self, train_data):
        self.z[0][0] = 1
        self.z[0][1:] = train_data
        for i in range(len(self.structure) - 2):
            self.y[i + 1][1:] = self.weight[i] @ self.z[i]
            self.z[i + 1] = self.actfunc(self.y[i + 1])
        self.y[-1] = self.weight[-1] @ self.z[-2]
        self.z[-1] = self.actfunc(self.y[-1])

    # バックプロパゲーション
    def backpropagation(self, train_data, train_label):
        #学習率の変更
        self.__fit_train_ratio(train_label, self.z[-1])
        # out layer to middle layer
        tmp = self.diffact(self.y[-1]) * self.diffcost(train_label, self.z[-1])
        diff = vmath.vvmat(self.z[-2], tmp)
        self.weight[-1] -= self.train_ratio * diff.T
        # middle layer to input layer
        for i in range(len(self.structure) - 2):
            weight = (self.weight[-i - 1].T[1:]).T
            z = (self.z[-i - 2].T[1:]).T
            tmp = self.diffact(z) * (weight.T @ tmp)
            diff = vmath.vvmat(self.z[-i - 3], tmp)
            self.weight[-i - 2] -= self.train_ratio * diff.T

    def __fit_train_ratio(self, train_label, ans):
        if self.costfunc(train_label, ans) < 0.5:
            self.train_ratio = 0.1
        elif self.costfunc(train_label, ans) < 0.1:
            self.train_ratio = 0.01

    # 学習
    def train(self, train_data, train_label):
        for i in range(len(train_data)):
            self.forwordpropagation(train_data[i])
            self.backpropagation(train_data[i], train_label[i])

    # テスト
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