import activationfunction as af
import makedata as md
import nural_network as nn

layer = [3, 2, 1]
epoch = 25

# set data
trainData = md.dset(epoch)
testData = md.dset(10)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(layer)
# 学習
# print(orNN.z)
# print(orNN.y)
# print(orNN.weight)
count = 500
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData,testData,"xivier","sigmoid","rss")
for i in range(count):
    orNN.train()
    orNN.test()
