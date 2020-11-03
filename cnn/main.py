import sys

sys.path.append('./dnn')
sys.path.append('./dataset')
import cnn_analysistool as catool
import neural_network as nne
import mnist

(train_data, train_label), (test_data, test_label) = mnist.load_data()
