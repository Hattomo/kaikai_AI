import sys

import numpy as np

class Normalization_Layer:

    def forwardpropagation(self, data, n_method="batch_normalization"):
        if n_method == "batch_normalization":
            return self.__batch_normalization(data)
        sys.stdout.write("Error: The normalization method is not found\n")
        sys.exit(1)

    def backpropagation(self,error):
        return error

    def __batch_normalization(self, data):
        # Batch Normalization
        data = (data - data.mean()) / (data.std() + 1e-9)
        if not (data.max() == 0 and data.min() == 0):
            data = data * (2 / (data.max() - data.min()))
            data -= (data.max() - 1)
        else:
            print("Warning👻 : all data is 0 in normalization layer")
        return data
