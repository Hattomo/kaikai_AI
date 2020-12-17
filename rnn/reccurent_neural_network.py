import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./shared')
import activationfunction as af
import costfunction as cf
import neural_network as nn

class Reccurent_Neural_Network():

    def __init__(self, structure, actfunc="sigmoid", costfunc="rss"):
        self.structure = structure
        self.weight = self.set_weight(self.structure)
        self.reccurent_weight = self.set_reccurent_weight(self.structure)
        self.actfunc, self.diffact = af.set_actfunc(actfunc)
        self.costfunc, self.diffcost = cf.set_costfunc(costfunc)
        self.layerstocker = list()
        self.errorstocker = list()

    # set all weight
    def set_weight(self, structure):
        weight_num = len(structure)
        weight_layer = list()
        for i in range(weight_num - 2):
            weight_layer.append(np.ones([structure[i + 1] - 1, structure[i]]))
        weight_layer.append(np.ones([structure[-1], structure[-2]]))
        return weight_layer

    # set reccurent weight
    def set_reccurent_weight(self, structure):
        weight_num = len(structure)
        weight_layer = list()
        for i in range(weight_num - 2):
            weight_layer.append(np.ones([structure[i + 1] - 1, structure[i + 1]]))
        return weight_layer

    # override
    def forwardpropagation(self, train_data):
        # ひとつ前のデータ
        self.z_1, self.y_1 = self.set_layer(self.structure)
        # 今回のデータ
        self.z, self.y = self.set_layer(self.structure)
        # input layer
        self.z[0][1:] = train_data
        # middle layer
        for i in range(len(self.structure) - 2):
            self.y[i + 1][1:] = self.z[i] @ self.weight[i].T + self.z_1[i] @ self.reccurent_weight[i].T
            self.z[i + 1] = self.actfunc(self.y[i + 1])
        # output layer
        self.y[-1] = self.z[-2] @ self.weight[-1].T
        self.z[-1] = self.actfunc(self.y[-1])

        # set all layer
    def set_layer(self, structure):
        layer_num = len(structure)
        znet, ynet = list(), list()
        for i in range(layer_num):
            x, y = np.ones([structure[i]]), np.ones([structure[i]])
            znet.append(x)
            ynet.append(y)
        return znet, ynet

    def backpropagation(self, train_label):
        train_ratio = 0.1
        # ひとつ前のエラーを格納
        self.error_1 = self.set_error(self.structure)
        # 現在のエラーを格納
        self.error = self.set_error(self.structure)
        # バックプロパゲーション
        self.error[-1] = self.diffact(self.y[-1]) * self.diffcost(train_label, self.z[-1])
        self.weight[-1] -= train_ratio * self.vvmat(self.z[-2], self.error[-1])
        # middle layer to input layer
        for i in range(len(self.structure) - 2):
            self.error[-i] = self.diffact(self.z[-i - 2][1:]) * (self.weight[-i - 1][:, 1:] @ self.error[-i + 1][1:])
            self.weight[-i - 2] -= train_ratio * self.vvmat(self.z[-i - 3],
                                                            self.error_1[-i + 1][1:] + self.error[-i + 1][1:])

    # エラーを格納する
    def set_error(self, structure):
        error = list()
        for i in range(len(structure)):
            error.append(np.zeros(structure[i]))
        return error

    def vvmat(self, a, b):
        mat = np.zeros([len(b), len(a)])
        for i in range(len(b)):
            for j in range(len(a)):
                mat[i][j] = b[i] * a[j]
        return mat