# write kai 10/23
# Accuracy aaverage 0.75
import activationfunction as af
import makedata as md
import nural_network as nn

layer = [3,3,1]

# main
# set data
trainData = md.dset(100)
testData = md.dset(20)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(layer)
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData,"xivier","sigmoid","RSS")
print(orNN.alllayer)
# 学習
# orNN.train()
# テスト
# orNN.test(testData)
