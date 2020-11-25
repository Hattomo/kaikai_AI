import sys

import numpy as np

sys.path.append('./dnn')
import dsetting

class GradientDesend():
    def __init__(self):
        self.train_ratio = 0.1

    def optimaize(self, error, cost):
        self.__fit_train_ratio(cost)
        return self.train_ratio * error

    def __fit_train_ratio(self,cost):
        if cost < 0.01:
            self.train_ratio = 0.001
        elif cost < 0.1:
            self.train_ratio = 0.01
        elif cost < 0.5:
            self.train_ratio = 0.1

class Adam():
    def __init__(self,structure):
        self.m = dsetting.set_weight(structure,"he") 
        self.v = dsetting.set_weight(structure,"he")
        self.beta1 = np.full(len(structure)-1,0.9)
        self.beta2 = np.full(len(structure)-1,0.999)
        self.epsiron = 10e-8
        self.alpa = 0.001

    def optimaize(self, error, cnt, layer):
        self.m[layer] = (self.beta1[layer] * self.m[layer]) + (1 - self.beta1[layer]) * error
        self.v[layer] = self.beta2[layer] * self.v[layer] + (1 - self.beta2)[layer] * (error ** 2)
        m_var = self.m[layer] / (1 - self.beta1[layer] ** cnt)
        v_var = self.v[layer] / (1 - self.beta2[layer] ** cnt)
        return self.alpa * m_var / np.sqrt(self.v[layer]+self.epsiron)
