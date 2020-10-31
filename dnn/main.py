import activationfunction as af
import makedata as md
import neural_network as nn

layer = [3, 3, 1]
epoch = 25

# set data
trainData = md.dset(epoch)
testData = md.dset(10)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(layer)
# 学習
count = 200
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData,testData,"xivier","sigmoid","rss")
for i in range(count):
    orNN.train()
    orNN.test()
