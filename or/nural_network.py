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
        self.alllayer[0][0] = 1
        self.alllayer[0][1:] = x
        for i in range(len(self.layer)-2):
            self.alllayer[2*i+1] = np.dot(self.alllayer[2*i],self.allweight[i])
            self.alllayer[2*i+2] = self.actfunc(self.alllayer[2*i+1])
        self.alllayer[-1] = np.dot(self.alllayer[-2],self.allweight[-1][0])
    # バックプロパゲーション
    def backpropagation(self,x,y):
        diff = (y-self.alllayer[-1])
        for i in range(len(self.layer),0):
            self.allweight[i-1] += self.train_ratio*diff*self.alllayer[i-1]
            diff = (y-self.y2[0])*self.alllayer[0][1:]*self.difffunc(self.y1[1:])*x
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
