import activationfunction as af
import makedata as md
import neural_network as nn
import analysistool as atool

layer = [3, 4, 4, 1]
epoch = 30

# set data
trainData = md.dset("xor",epoch)
testData = md.dset("xor",20)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(layer)
# 学習
count = 1000
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData, testData, "xivier", "sigmoid", "rss")
for i in range(count):
    orNN.train()
    orNN.test()
atool.draw(orNN.cost)