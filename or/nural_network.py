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
        self.alllayer = [setting.ynet(self.layer), setting.znet(self.layer)]
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
            self.actfunc = [af.sigmoid, af.diffsigmoid]
        elif actfunc == "tanh":
            self.actfunc = [af.tanh, af.difftanh]
        elif actfunc == "ReLU":
            self.actfunc = [af.ReLU, af.diffReLU]
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
        self.alllayer[1][0][0] = 1
        self.alllayer[1][0][1:] = x
        for i in range(len(self.layer) - 1):
            self.alllayer[0][i] = self.alllayer[1][i] @ np.transpose(self.allweight[i])
            self.alllayer[1][i+1] = self.actfunc[0](self.alllayer[0][i])
    # バックプロパゲーション
    def backpropagation(self,x,y):
        # out layer to middle layer
        tmp = (y - self.alllayer[0][-2]) * self.actfunc[1](self.alllayer[0][-1])
        diff = self.alllayer[1][-1].reshape(1,1) * np.transpose(tmp)
        self.allweight[-1] -= self.train_ratio * diff
        # middle layer to input layer
        for i in range(len(self.layer) - 2):
            tmp = self.actfunc[1](self.alllayer[1][-i-2]) * (self.allweight[-i-1] @ np.transpose(tmp))
            diff = tmp * self.alllayer[1][-i-1]
            self.allweight[-1-i] -= self.train_ratio * diff
            
    # 学習
    def train(self):
        length = len(self.data)
        for i in range(50):
            for j in range(length):
                if j < 100:
                    self.train_ratio = 1
                else :
                    self.train_ratio = 0.1
                self.forwordpropagation(self.data[i][:-1])
                self.backpropagation(self.data[i][:-1], self.data[i][-1])
    # テスト
    def test(self,testdata):
        count = 0
        length = len(testdata)
        for i in range(length):
            self.forwordpropagation(testdata[i][:-1])
            if self.alllayer[0][-1] >= 0.5 and testdata[i][-1]==1:
                # print("ok")
                count += 1
            elif self.alllayer[0][-1] < 0.5 and testdata[i][-1]==0:
                # print("ok")
                count += 1
            else :
                pass
                # print("bad")
        print(count/length)

    
    def wprint(self):
        print(self.imWeight)
        print(self.moWeight)
