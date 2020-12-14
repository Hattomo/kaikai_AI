import numpy as np

import reccurent_neural_network as rnn

structure = [3,3,2]
myrnn = rnn.Reccurent_Neural_Network(structure)
train_data = np.ones(2)
error = np.ones(2)
myrnn.forwardpropagation(train_data)
myrnn.backpropagation(error)
print()