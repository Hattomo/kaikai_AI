import sys

import numpy as np

sys.path.append('./dnn')
sys.path.append('./dataset')
import analysistool as atool
import cnn_analysistool as catool
import convolution_layer as cl
import csetting
import convolution_neural_network as cnn
import fully_connenct_layer as fc
import neural_network as nn
import normalization_layer as nl
import pooling_layer as pl
import mnist
import dataset

data_name = "cnn_ex"
datasetsize = 8
batch = 4
trainData, trainLabel = dataset.imgtrain(data_name,datasetsize,batch)
testData, testLabel= dataset.imgtest(data_name,testsize=4)
# 各層の生成
conv = cl.Convolution_Layer(in_channel=1, out_channel=8, ksize=3, pad=1)
pool = pl.Pooling_Layer(pooling_size=[2, 2])
norm = nl.Normalization_Layer()
fullc = fc.Fully_Connect_Layer([32 + 1, 10, 4])
mycnn = cnn.Convolution_Neural_Network([conv,pool,norm,fullc])

epoch = 1000
for i in range(epoch):
    mycnn.train(trainData,trainLabel)
    mycnn.test(testData,testLabel)
