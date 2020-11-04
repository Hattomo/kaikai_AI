import sys

sys.path.append("./dataset")
import activationfunction as af
import analysistool as atool
import neural_network as nn
import files
import logic_circuit as lc

layer = [795, 350, 10]
epoch = 30
logic = "or"
# set data
trainData = lc.dset("mnist_train", 1)
testData = lc.dset("mnist_test", 1)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(layer)
# 学習
count = 1
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData, testData, "xivier", "sigmoid", "rss")
for i in range(count):
    orNN.train()
    orNN.test()
atool.draw(orNN.cost)
atool.tdchart(orNN)
files.save(orNN)