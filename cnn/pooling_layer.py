import numpy as np

class Pooling_Layer:

    def __init__(self, data, pooling_size, pooling_method="max-pooling"):
        self.data = data
        self.pooling_method = pooling_method
        self.pooling_size = pooling_size

    def pooling(self):
        channel = len(self.data)
        height = len(self.data[0])
        width = len(self.data[0][0])
        if width % self.pooling_size[0] != 0 or height % self.pooling_size[1] != 0:
            sys.stdout.write("Error: The pooling_size is not right\n")
            sys.exit(1)
        out_height = int(height / self.pooling_size[1])
        out_width = int(width / self.pooling_size[0])
        out = np.zeros([channel, out_width, out_width])
        if self.pooling_method == "max-pooling":
            for h in range(channel):
                for i in range(out_height):
                    for j in range(out_width):
                        out[h][i][j] = np.max(self.data[h][i:i + self.pooling_size[1]].T[j:j + self.pooling_size[0]].T)
        elif self.pooling_method == "mean-pooling":
            for h in range(channel):
                for i in range(out_height):
                    for j in range(out_width):
                        out[h][i][j] = np.average(self.data[h][i:i + self.pooling_size[1]].T[j:j + self.pooling_size[0]].T)
        else:
            sys.stdout.write("Error: The pooling_method is not found\n")
            sys.exit(1)
        return out