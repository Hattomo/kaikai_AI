import sys
import io

import numpy as np

sys.path.append('./dataset')
sys.path.append('./tools')
sys.path.append('./shared')
import analysistool as atool
import convolution_layer as cl
import convolutional_neural_network as cnn
import cleaner
import dataset
import doc_maker
import fully_connenct_layer as fc
import normalization_layer as nl
import pooling_layer as pl
import mnist
import unbuffered

# output system set
s, timestamp, commitid, branchname = doc_maker.getdata("cnn")
stdout_stream = io.StringIO()
sys.stdout = unbuffered.Unbuffered(sys.stdout, stdout_stream)

# make data
data_name = "mnist"
datasetsize = 100
batch = 10
trainData, trainLabel, testData, testLabel = dataset.image_mnist(data_name, datasetsize, batch, testsize=8)
# generate each layer
conv1 = cl.Convolution_Layer(in_channel=1, out_channel=4, ksize=3, pad=1)
pool1 = pl.Pooling_Layer(pooling_size=[2, 2])
conv2 = cl.Convolution_Layer(in_channel=4, out_channel=8, ksize=3)
pool2 = pl.Pooling_Layer(pooling_size=[3, 3])
norm1 = nl.Normalization_Layer()
norm2 = nl.Normalization_Layer()
fullc = fc.Fully_Connect_Layer([128 + 1, 10, 9])
mycnn = cnn.Convolutional_Neural_Network([conv1, pool1, norm1, conv2, pool2, norm2, fullc])
# train and test
epoch = 500
for i in range(epoch):
    mycnn.train(trainData, trainLabel)
    mycnn.test(testData, testLabel)

# draw graph
doc_maker.docmaker(s, timestamp, stdout_stream.getvalue(), commitid, branchname)
atool.draw(fullc.cost, timestamp)
atool.accurancygraph(fullc.accurancy, timestamp)
atool.kernelmove(conv.move, timestamp)
cleaner.clean()
