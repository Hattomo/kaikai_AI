import sys
import activationfunction as af
import lossfunction
import setting
import numpy as np

class Neural_Network:
    def __init__(self,layer):
        self.layer = layer
        self.alllayer = None
        self.allweight = None
        self.actfunc = None
        self.lossfunc = None

    def model(self,data,w_method="unif",actfunc="sigmoid",lossfunc="RSS"):
        self.data = data
        self.alllayer = setting.lnet(self.layer)
        # 重みの初期化
        if w_method == "xivier":
            self.allweight = setting.wnet(self.layer,setting.xivier)
        elif w_method == "he":
            self.allweight = setting.wnet(self.layer,setting.he)
        elif w_method == "unif":
            self.allweight = setting.wnet(self.layer,setting.unif)
        else:
            sys.stdout.write("Error: The weight method is not found")
            sys.exit(0)
        # 活性化関数の初期化
        if actfunc == "sigmoid":
            self.actfunc = af.sigmoid
        elif actfunc == "tanh":
            self.actfunc = af.tanh
        elif actfunc == "ReLU":
            self.actfunc = af.ReLU
        else:
            sys.stdout.write("Error: The actfunc is not found")
            sys.exit(0)
        # 損失関数の初期化
        if lossfunc == "RSS":
            self.lossfunc = lossfunction.RSS
        else:
            sys.stdout.write("Error: The lossfunc is not found")
            sys.exit(0)
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
