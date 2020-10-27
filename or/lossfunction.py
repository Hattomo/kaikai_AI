import math

def RSS(self, data):
        cost=0
        length = len(data)
        for i in range(length):
            self.forwordpropagation(data[i][:-1])
            cost+= (self.y2[0] - data[i][-1])**2
        print(cost)
        return cost