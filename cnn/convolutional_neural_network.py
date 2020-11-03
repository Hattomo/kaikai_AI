sys.path.append('./dnn')
import neural_network as nn

class Convolutional_Neural_Network:

    def __init__(self, layer):
        self.nn = nn.Neural_Network()
    
    def model(data,testdata,filter,stride,padding_method)
        self.data = data
        self.testdata = testdata
        self.filter = filer
        self.stride = stride
        self.padding_method = padding_method

    def conv(x,stride,filter,padding_method):
        x_size = len(x)
        filter_size = len(filter)
        #padding
        if padding_method == "valid-padding":
            x = x
            padding_size = 0
            x_size += padding_size
        else:
            sys.stdout.write("Error: The padding method is not found\n")
            sys.exit(1)
        # make output array
        if (x_size-filter_size) == (x_size-filter_size)/stride * stride:
            out_size =int( (x_size-filter_size)/stride+1 )
            out = np.zeros([out_size,out_size])
        else:
            sys.stdout.write("Error: The stride is not right\n")
            sys.exit(1)
        #convolution
        for i in range(out_size):
            for j in range(out_size):
                # print( x[i:i+filter_size].T[j:j+filter_size].T )
                y = (x[i:i+filter_size].T[j:j+filter_size].T) @ filter
                z = sigmoid(y)
                out[i][j] = np.sum(z)
        return out
    def pooling(x,pooling_method):
        x_size = len(x)
        if pooling_method == "max-pooling":
            pass
        elif pooling_method == "mean-pooling":
            pass