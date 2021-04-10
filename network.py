import numpy as np
from functions import ReLU
from layer import *


class NeuralNetwork:
    def __init__(self, input_size, num_of_layers, num_of_classes, activation=ReLU, policy="constant"):

            if policy=="constant" or num_of_layers == 1:
                self.layers = [Layer(input_size,input_size) for i in range(num_of_layers-1)] + [SoftMaxLayer(input_size,num_of_classes)]

            else:
                self.layers = [Layer(input_size, 6)] + [Layer(2 * (i + 2), 2 * (i + 3)) for i in range(1, num_of_layers // 2)] \
                          + [Layer(2 * (i + 3), 2 * (i + 2)) for i in range(num_of_layers // 2 - 1 , 0, -1)] \
                          + [SoftMaxLayer(6, num_of_classes)]


    def forward_pass(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)


    def backward_pass(self):
        pass





#if __name__ == '__main__':
