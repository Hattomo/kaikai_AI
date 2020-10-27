import activationfunction as af
import setting as set
import numpy as np

class Nural_Network_3:
    def __init__(self,input_node,middle_node,output_node):
        self.input_node = input_node
        self.middle_node = middle_node
        self.output_node = output_node
        self.train_ratio = 0.1
        self.data = np.ones(0)
        self.imWeight = np.ones(0)
        self.moWeight = np.ones(0)
        self.y1 = set.layer(middle_node)
        self.h1 = self.y1
        self.y2 = set.layer(output_node)
        self.func = af.sigmoid
    def set(self,data,weight_num,func_num):
        self.data = data
        if weight_num == "xivier":
            self.imWeight = set.wset_xivier(self.input_node,self.middle_node-1)
            self.moWeight = set.wset_xivier(self.middle_node,self.output_node)
        elif weight_num == "he":
            self.imWeight = set.wset_he(self.input_node,self.middle_node-1)
            self.moWeight = set.wset_he(self.middle_node,self.output_node)
        else :
            self.imWeight = set.wset_unif(self.input_node,self.middle_node-1)
            self.moWeight = set.wset_unif(self.middle_node,self.output_node)
        
        self.func = af.sigmoid
        self.difffunc = af.msigmoid
    # フォワードプロパゲーション
    def forwordpropagation(self,x):
        self.y1[1:] = np.dot(self.imWeight,x)
        self.h1 = self.func(self.y1)
        self.y2 = np.dot(self.moWeight,self.h1)
    # バックプロパゲーション
    def backpropagation(self,x,y):
        # 中間層と出力層間の重みの更新
        diff = (y-self.y2[0])*self.h1
        self.moWeight += self.train_ratio*diff
        # 入力層と中間層間の重み更新
        diff = (y-self.y2[0])*self.moWeight[0][1:]*self.difffunc(self.y1[1:])*x
        self.imWeight += self.train_ratio*diff
    # 学習
    def train(self):
        length = len(self.data)
        for i in range(length):
            if i < 300:
                self.train_ratio = 0.1
            else :
                self.train_ratio = 0.01
            self.forwordpropagation(self.data[i][:-1])
            self.backpropagation(self.data[i][:-1],self.data[i][-1])
    # テスト
    def test(self,testdata):
        count = 0
        length = len(testdata)
        for i in range(length):
            self.forwordpropagation(testdata[i][:-1])
            if self.y2 >= 0.5 and testdata[i][-1]==1:
                # print("ok")
                count += 1
            elif self.y2 < 0.5 and testdata[i][-1]==0:
                # print("ok")
                count += 1
            else :
                pass
                # print("bad")
        print(count/length)
    def wprint(self):
        print(self.imWeight)
        print(self.moWeight)
