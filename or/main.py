import activationfunction as af
import makedata as md
import nural_network as nn

layer = [3,3,1]
epoch = 25

# main
# set data
trainData = md.dset(epoch)
testData = md.dset(1)
# ニューラルネットワークの生成
orNN = nn.Neural_Network(layer)
# ニューラルネットワークのトレーニングデータ、レイヤー、重み番号、活性化関数番号の設定,cost func
orNN.model(trainData,testData,"xivier","sigmoid","RSS")
# orNN.forwordpropagation(trainData[0][:-1])
# 学習
# print(orNN.weight)
# print(orNN.z)
# print(orNN.y)
orNN.train()
# テスト
orNN.test(testData)
# print(orNN.alllayer)
# print(orNN.allweight)
sum = 0
num = 25
for i in range(num):
    orNN.model(trainData,testData,"xivier","sigmoid","RSS")
    orNN.train()
    orNN.test(testData)
    sum += orNN.accuracy
print(sum/num)

