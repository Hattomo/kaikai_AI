# write kai 10/23
# Accuracy aaverage 0.75
import activationfunction as af
import makedata as md
import nural_network as nn

TRAIN_RATIO = 0.01
INPUT_NODE = 2
MIDDLE_NODE = 3
OUTPUT_NODE = 1

class Nural_Network(nn.Nural_Network_3):
    def __init__(self,input_node,middle_node,output_node):
        super().__init__(input_node,middle_node,output_node)



# main
# set data
trainData = md.dset(100)
testData = md.dset(20)
# ニューラルネットワークの生成
orNN = Nural_Network(INPUT_NODE,MIDDLE_NODE,OUTPUT_NODE)
# ニューラルネットワークのトレーニングデータ、重み番号、活性化関数番号の設定
orNN.set(trainData,1,1)
# 学習
orNN.train()
# テスト
orNN.test(testData)
'''
・問題点
学習率、重みの初期値、層の数、ノードの数が適切でない可能性がある。
・解決策
NN(2-3-3-1)の実装
・課題
重みの初期値の最適化(理論)、効率化
'''
